import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from solo.utils.misc import generate_2d_sincos_pos_embed
from solo.methods.base import BaseMethod
import omegaconf

class CrossAttention(nn.Module):
    """Cross-attention module for incorporating temporal context."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Query projections for target frame
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # Key, Value projections for context frame
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape
        
        # Project query from target frame
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Project key and value from context frame
        k = self.k(context).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(context).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TemporalBlock(nn.Module):
    """Vision Transformer Block with support for cross-attention."""
    def __init__(
        self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_cross_attention=False
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        # Use self-attention or cross-attention depending on the flag
        if use_cross_attention:
            self.attn = CrossAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                attn_drop=attn_drop, proj_drop=drop
            )
        else:
            # Use original self-attention from timm
            self.attn = nn.MultiheadAttention(
                dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True
            )
        
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
        
        self.use_cross_attention = use_cross_attention

    def forward(self, x, context=None):
        if self.use_cross_attention and context is not None:
            x = x + self.attn(self.norm1(x), self.norm1(context))
        else:
            # Standard self-attention
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalMAEDecoder(nn.Module):
    def __init__(
        self, in_dim, embed_dim, depth, num_heads, num_patches, patch_size, 
        context_dim=None, cross_attention_layers=2, mlp_ratio=4.0
    ) -> None:
        super().__init__()

        self.num_patches = num_patches
        context_dim = context_dim or in_dim  # Default to same dimension if not specified

        # Decoder embedding for masked tokens
        self.decoder_embed = nn.Linear(in_dim, embed_dim, bias=True)
        
        # Context projection for temporal information
        self.context_projection = nn.Linear(context_dim, embed_dim, bias=True)

        # Mask token embedding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Fixed sin-cos positional embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        # Build decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        # First add cross-attention blocks for temporal context
        for i in range(cross_attention_layers):
            self.decoder_blocks.append(
                TemporalBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    use_cross_attention=True
                )
            )
        
        # Then add standard self-attention blocks
        for i in range(depth - cross_attention_layers):
            self.decoder_blocks.append(
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
            )

        # Final normalization and prediction head
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * 3, bias=True)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embeddings with sin-cos embedding
        decoder_pos_embed = generate_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize mask token with small random values
        nn.init.normal_(self.mask_token, std=0.02)

        # Initialize linear layers and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Xavier initialization for linear layers
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore, context=None):
        """
        Forward pass of the temporal decoder.
        
        Args:
            x: Embedded tokens from the target frame (includes only visible tokens)
            ids_restore: Indices to restore the original order of patches
            context: Embedded tokens from the context frame (optional)
            
        Returns:
            Reconstructed pixel values for masked patches
        """
        # Embed tokens from target frame
        x = self.decoder_embed(x)
        
        # Process context if provided
        if context is not None:
            context = self.context_projection(context)

        # Append mask tokens to visible tokens sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        
        # Unshuffle to recover original patch ordering
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # Add positional embeddings
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for i, block in enumerate(self.decoder_blocks):
            if isinstance(block, TemporalBlock) and context is not None:
                x = block(x, context)
            else:
                x = block(x)

        # Final normalization
        x = self.decoder_norm(x)

        # Predict pixel values
        x = self.decoder_pred(x)

        # Remove cls token
        x = x[:, 1:, :]

        return x


class TemporalMAE(BaseMethod):
    """Temporal Masked Autoencoder for frame-to-frame prediction."""
    
    def __init__(self, cfg: omegaconf.DictConfig):
        """
        Implements Temporal MAE for frame-to-frame prediction.
        
        Args:
            cfg: Configuration object
        """
        super().__init__(cfg)
        
        # Check backbone is ViT
        assert "vit" in self.backbone_name, "TemporalMAE only supports ViT as backbone."
        
        # Extract MAE-specific parameters
        self.mask_ratio = cfg.method_kwargs.mask_ratio
        self.norm_pix_loss = cfg.method_kwargs.norm_pix_loss
        
        # Extract temporal-specific parameters
        self.delta_max = cfg.method_kwargs.get("delta_max", 15)
        
        # Get ViT configuration from backbone (now available via self)
        self._vit_embed_dim = self.backbone.pos_embed.size(-1)
        default_patch_size = 14 if self.backbone_name == "vit_huge" else 16
        self._vit_patch_size = self.backbone_args.get("patch_size", default_patch_size)
        self._vit_num_patches = self.backbone.patch_embed.num_patches
        
        # Create temporal MAE decoder
        decoder_embed_dim = cfg.method_kwargs.decoder_embed_dim
        decoder_depth = cfg.method_kwargs.decoder_depth
        decoder_num_heads = cfg.method_kwargs.decoder_num_heads
        cross_attention_layers = cfg.method_kwargs.get("cross_attention_layers", 2)
        
        self.decoder = TemporalMAEDecoder(
            in_dim=self.features_dim,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            num_patches=self._vit_num_patches,
            patch_size=self._vit_patch_size,
            context_dim=self.features_dim,
            cross_attention_layers=cross_attention_layers,
            mlp_ratio=4.0,
        )
    
    def forward(self, X, compute_metrics=False):
        """
        Forward pass for Temporal MAE.
        
        Args:
            X: Input data, can be:
               - A single tensor (standard MAE mode)
               - A list of two tensors [current_frame, future_frame] (temporal mode)
            compute_metrics: If True, will compute and return masking/prediction data
                            even during evaluation (for metric calculation)
               
        Returns:
            Dict with model outputs
        """
        out = {}
        
        # Handle temporal mode (list of two frames)
        if isinstance(X, list) and len(X) >= 2:
            current_frame = X[0]
            future_frame = X[1]
            
            # Convert to channels_last if needed
            if not self.no_channel_last:
                current_frame = current_frame.to(memory_format=torch.channels_last)
                future_frame = future_frame.to(memory_format=torch.channels_last)
            
            if self.training or compute_metrics:
                # Extract features from current frame without masking
                current_feats = self.backbone.forward_features(current_frame)
                
                # Apply masking to future frame - use our custom masking approach
                # since the backbone doesn't accept mask_ratio directly
                
                # 1. Get patch embeddings (forward_features includes the cls token)
                future_feats = self.backbone.forward_features(future_frame)
                patch_feats = future_feats[:, 1:, :]  # Remove CLS token
                
                # 2. Create mask
                B = patch_feats.shape[0]
                N = patch_feats.shape[1]  # number of patches
                
                # Create a random mask (1 = keep, 0 = mask)
                noise = torch.rand(B, N, device=patch_feats.device)  # noise in [0, 1]
                
                # Sort noise for each sample
                ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is mask
                ids_restore = torch.argsort(ids_shuffle, dim=1)  # restore the original order
                
                # Keep the first (1-mask_ratio) patches, mask the remaining
                len_keep = int(N * (1 - self.mask_ratio))
                mask = torch.ones(B, N, device=patch_feats.device)
                mask[:, :len_keep] = 0  # 0 is keep, 1 is mask
                
                # Unshuffle to get the binary mask
                mask = torch.gather(mask, dim=1, index=ids_restore)
                
                # Use only the visible tokens (where mask=0) for the decoder input
                # First, shuffle the patch features according to ids_shuffle
                patch_feats_shuffled = torch.gather(
                    patch_feats, 
                    dim=1,
                    index=ids_shuffle.unsqueeze(-1).repeat(1, 1, patch_feats.shape[-1])
                )
                
                # Then take only the first len_keep patches (visible ones)
                visible_patches = patch_feats_shuffled[:, :len_keep, :]
                
                # Add back the CLS token
                visible_patches_with_cls = torch.cat([future_feats[:, :1, :], visible_patches], dim=1)
                
                # Use the decoder to predict masked patches in future frame
                # using current frame as context
                pred = self.decoder(visible_patches_with_cls, ids_restore, context=current_feats)
                
                out.update({
                    "mask": mask, 
                    "pred": pred, 
                    "target_frame": future_frame
                })
            else:
                # During inference, just get features
                current_feats = self.backbone(current_frame)
                
            # Add classifier outputs if available
            if self.classifier is not None:
                logits = self.classifier(current_feats.detach())
                out.update({"logits": logits})
                
            out.update({"feats": current_feats})
            
        # Standard MAE mode (single frame)
        else:
            if not self.no_channel_last:
                X = X.to(memory_format=torch.channels_last)

            if self.training or compute_metrics:
                # 1. Get patch embeddings (forward_features includes the cls token)
                feats = self.backbone.forward_features(X)
                patch_feats = feats[:, 1:, :]  # Remove CLS token
                
                # 2. Create mask
                B = patch_feats.shape[0]
                N = patch_feats.shape[1]  # number of patches
                
                # Create a random mask (1 = keep, 0 = mask)
                noise = torch.rand(B, N, device=patch_feats.device)  # noise in [0, 1]
                
                # Sort noise for each sample
                ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is mask
                ids_restore = torch.argsort(ids_shuffle, dim=1)  # restore the original order
                
                # Keep the first (1-mask_ratio) patches, mask the remaining
                len_keep = int(N * (1 - self.mask_ratio))
                mask = torch.ones(B, N, device=patch_feats.device)
                mask[:, :len_keep] = 0  # 0 is keep, 1 is mask
                
                # Unshuffle to get the binary mask
                mask = torch.gather(mask, dim=1, index=ids_restore)
                
                # Use only the visible tokens (where mask=0) for the decoder input
                # First, shuffle the patch features according to ids_shuffle
                patch_feats_shuffled = torch.gather(
                    patch_feats, 
                    dim=1,
                    index=ids_shuffle.unsqueeze(-1).repeat(1, 1, patch_feats.shape[-1])
                )
                
                # Then take only the first len_keep patches (visible ones)
                visible_patches = patch_feats_shuffled[:, :len_keep, :]
                
                # Add back the CLS token
                visible_patches_with_cls = torch.cat([feats[:, :1, :], visible_patches], dim=1)
                
                # Use the decoder to predict
                pred = self.decoder(visible_patches_with_cls, ids_restore)
                
                out.update({"mask": mask, "pred": pred})
            else:
                feats = self.backbone(X)

            if self.classifier is not None:
                logits = self.classifier(feats.detach())
                out.update({"logits": logits})
                
            out.update({"feats": feats})
            
        return out
        
    def extract_patches(self, imgs, patch_size):
        """
        Extract patches from images for reconstruction loss.
        
        Args:
            imgs: Images tensor [B, C, H, W]
            patch_size: Size of patches
            
        Returns:
            Patches tensor [B, N, P*P*C]
        """
        B, C, H, W = imgs.shape
        patches = imgs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, -1, C * patch_size ** 2)
        return patches
    
    def training_step(self, batch, batch_idx):
        """
        Training step for Temporal MAE.
        
        Args:
            batch: Input batch, can be in different formats
            batch_idx: Index of the batch (required by Lightning)
            
        Returns:
            Loss tensor
        """
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                # Format: [indexes, views, targets]
                indexes, all_views, targets = batch
            elif len(batch) == 2:
                # Format: [indexes, views]
                indexes, all_views = batch
            else:
                # Assume it's the views
                all_views = batch[0] if isinstance(batch[0], (list, tuple)) else batch
        else:
            all_views = batch
        
        # Make sure all_views is a list
        if not isinstance(all_views, (list, tuple)):
            all_views = [all_views]
        
        # Check if batch contains enough crops for temporal processing
        if len(all_views) < 2:
            # Return dummy loss if not enough views
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Use first two crops as temporal pair
        x_t = all_views[0]  # Current frame
        x_t_plus_delta = all_views[1]  # Future frame
        
        # Forward pass with both frames
        out = self.forward([x_t, x_t_plus_delta])
        
        # Extract patches from target frame for loss computation
        target_patches = self.extract_patches(x_t_plus_delta, self._vit_patch_size)
        
        # Compute per-patch mean and std for normalization if enabled
        if self.norm_pix_loss:
            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            target_patches = (target_patches - mean) / (var + 1e-6)**.5
        
        # Compute reconstruction loss (MSE) on masked patches only
        mask = out["mask"]
        pred = out["pred"]
        
        loss = (pred - target_patches) ** 2
        loss = loss.mean(dim=-1)  # Mean over patch dimension
        loss = (loss * mask).sum() / mask.sum()  # Mean over masked patches
        
        # Log the training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss