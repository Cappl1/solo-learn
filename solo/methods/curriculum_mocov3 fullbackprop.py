import math
from typing import Any, Dict, List, Sequence, Tuple

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.mocov3 import mocov3_loss_func
from solo.methods.mocov3 import MoCoV3
from solo.utils.misc import gather


class CurriculumMoCoV3(MoCoV3):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements MoCo V3 with Curriculum Learning via reconstruction.
        
        Extends MoCo V3 with a reconstruction branch (either MAE or JEPA style).
        Uses reconstruction error to dynamically weight the contrastive loss.
        
        Extra cfg settings:
            method_kwargs:
                curriculum_type (str): either "mae" or "jepa"
                curriculum_strategy (str): "binary", "exponential", or "bands"
                curriculum_warmup_epochs (int): number of epochs to warm up the curriculum
                curriculum_weight (float): weight for the reconstruction loss
                reconstruction_masking_ratio (float): ratio of tokens to mask/predict
        """
        super().__init__(cfg)
        
        # Get curriculum learning parameters
        self.curriculum_type = cfg.method_kwargs.get("curriculum_type", "mae")  # "mae" or "jepa"
        self.curriculum_strategy = cfg.method_kwargs.get("curriculum_strategy", "exponential")  # "binary", "exponential", or "bands"
        self.curriculum_warmup_epochs = cfg.method_kwargs.get("curriculum_warmup_epochs", 10)
        self.curriculum_weight = cfg.method_kwargs.get("curriculum_weight", 1.0)
        self.masking_ratio = cfg.method_kwargs.get("reconstruction_masking_ratio", 0.75)
        
        print(f"Using curriculum strategy: {self.curriculum_strategy}")
        
        # Check if we're using a ViT backbone
        self.is_vit = 'vit' in self.backbone_name
        
        if self.is_vit:
            # Get transformer parameters
            if hasattr(self.backbone, "patch_embed"):
                self.patch_size = self.backbone.patch_embed.patch_size[0]
                embed_dim = self.backbone.embed_dim
            else:
                # Try alternative attribute names for different ViT implementations
                self.patch_size = getattr(self.backbone, "patch_size", 16)
                embed_dim = getattr(self.backbone, "embed_dim", 768)
            
            # We need to know the image size - for now we'll assume 224
            self.img_size = 224
            self.num_patches = (self.img_size // self.patch_size) ** 2
            
            # Create a decoder based on curriculum type
            if self.curriculum_type == "mae":
                # MAE-style decoder is lightweight
                self.decoder = self._build_mae_decoder(embed_dim)
                # Create a mask token
                self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                nn.init.normal_(self.mask_token, std=0.02)
            else:  # JEPA style
                # JEPA-style predictor predicts latent representations from context
                self.predictor = self._build_jepa_predictor(embed_dim)
        else:
            # For ResNet/CNN backbones, use our proper decoder for 224x224 images
            self.decoder = self._build_resnet_decoder()
    
    def _build_mae_decoder(self, embed_dim: int) -> nn.Module:
        """Build a lightweight decoder for MAE-style reconstruction."""
        decoder_dim = embed_dim // 2
        decoder_depth = 4
        
        decoder_blocks = []
        for _ in range(decoder_depth):
            decoder_blocks.append(
                nn.TransformerDecoderLayer(
                    d_model=decoder_dim,
                    nhead=8,
                    dim_feedforward=decoder_dim * 4,
                    batch_first=True,
                )
            )
        
        decoder = nn.Sequential(
            nn.Linear(embed_dim, decoder_dim),
            nn.GELU(),
            nn.TransformerDecoder(decoder_blocks[0], num_layers=decoder_depth),
            nn.Linear(decoder_dim, self.patch_size * self.patch_size * 3)
        )
        
        return decoder
    
    def _build_jepa_predictor(self, embed_dim: int) -> nn.Module:
        """Build a predictor for JEPA-style latent prediction."""
        predictor_dim = embed_dim
        predictor = nn.Sequential(
            nn.Linear(embed_dim, predictor_dim * 2),
            nn.LayerNorm(predictor_dim * 2),
            nn.GELU(),
            nn.Linear(predictor_dim * 2, predictor_dim)
        )
        
        return predictor
    
    def _build_resnet_decoder(self) -> nn.Module:
        """Build a decoder for ResNet features that reconstructs 224x224 images."""
        features_dim = self.features_dim  # Usually 512 for ResNet18
        
        class ResNetDecoder(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                
                # Calculate proper dimensions for reshape
                self.initial_shape = (256, 14, 14)  # 256 channels, 14x14 spatial size
                flat_dim = 256 * 14 * 14  # Calculate exact size needed
                
                # Initial projection from feature vector to exact size needed for reshape
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, flat_dim),
                    nn.BatchNorm1d(flat_dim),
                    nn.ReLU(inplace=True)
                )
                
                # Upsample from 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
                self.decoder = nn.Sequential(
                    # 14x14 -> 28x28
                    nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    
                    # 28x28 -> 56x56
                    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    # 56x56 -> 112x112
                    nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    
                    # 112x112 -> 224x224
                    nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    
                    # Final conv to get 3 channels
                    nn.Conv2d(32, 3, kernel_size=3, padding=1),
                    nn.Tanh()  # Output in [-1, 1] range to match normalized images
                )
            
            def forward(self, x):
                # Project feature vector
                batch_size = x.shape[0]
                x = self.fc(x)
                
                # Reshape to initial spatial dimensions
                x = x.view(batch_size, *self.initial_shape)
                
                # Decode to image
                x = self.decoder(x)
                
                return x
        
        return ResNetDecoder(features_dim)
    
    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config."""
        cfg = super(CurriculumMoCoV3, CurriculumMoCoV3).add_and_assert_specific_cfg(cfg)
        
        # Add curriculum-specific parameters with defaults if missing
        if "curriculum_type" not in cfg.method_kwargs:
            cfg.method_kwargs.curriculum_type = "mae"
            
        if "curriculum_strategy" not in cfg.method_kwargs:
            cfg.method_kwargs.curriculum_strategy = "exponential"
            
        if "curriculum_warmup_epochs" not in cfg.method_kwargs:
            cfg.method_kwargs.curriculum_warmup_epochs = 10
            
        if "curriculum_weight" not in cfg.method_kwargs:
            cfg.method_kwargs.curriculum_weight = 1.0
            
        if "reconstruction_masking_ratio" not in cfg.method_kwargs:
            cfg.method_kwargs.reconstruction_masking_ratio = 0.75
        
        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds decoder/predictor parameters to the parent's learnable parameters."""
        extra_params = super().learnable_params
        
        if self.curriculum_type == "mae" or not self.is_vit:
            extra_params.append({"name": "decoder", "params": self.decoder.parameters()})
        elif self.curriculum_type == "jepa":
            extra_params.append({"name": "predictor", "params": self.predictor.parameters()})
        
        return extra_params
    
    def _compute_mae_reconstruction(self, feats, mask, batch):
        """Compute MAE-style reconstruction and error.
        
        Args:
            feats: features from the backbone
            mask: boolean mask indicating which patches to reconstruct (True = masked)
            batch: the original batch for access to original images
            
        Returns:
            Tuple of (reconstruction loss, per-sample reconstruction errors)
        """
        B = feats.shape[0]
        
        # Get original images - using first view
        orig_images = batch[1][0]  # Shape: [B, 3, 224, 224]
        
        # Extract original image patches
        # For ViT, we need to extract patches matching the tokenization
        # First divide the image into patch_size x patch_size patches
        patches_h = patches_w = self.img_size // self.patch_size
        
        # Unfold the image into patches [B, 3, patches_h, patch_size, patches_w, patch_size]
        patches = orig_images.unfold(2, self.patch_size, self.patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)
        
        # Reshape to [B, patches_h*patches_w, 3*patch_size*patch_size]
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        orig_patches = patches.view(B, patches_h * patches_w, 3 * self.patch_size * self.patch_size)
        
        # If we have a CLS token, adjust our mask indexing
        cls_offset = 1 if hasattr(self.backbone, "cls_token") else 0
        
        # Replace masked tokens with mask token
        mask_tokens = self.mask_token.repeat(B, mask.sum(dim=1), 1)
        
        # Combine visible tokens with mask tokens
        x_full = feats.clone()
        x_full[mask] = mask_tokens.reshape(-1, feats.shape[-1])
        
        # Decode the full sequence
        pred_patches = self.decoder(x_full)
        
        # Compute loss only on masked positions
        # Skip the CLS token if present
        masked_pred = []
        masked_orig = []
        
        for i in range(B):
            # Get masked indices, adjusting for CLS token if needed
            mask_indices = mask[i, cls_offset:].nonzero().squeeze(1)
            
            if len(mask_indices.shape) == 0:  # Handle case of single masked patch
                mask_indices = mask_indices.unsqueeze(0)
            
            # Get predicted patches for masked positions
            if mask_indices.numel() > 0:  # Check if there are any masked patches
                masked_pred.append(pred_patches[i, mask_indices])
                # Get original patches at same positions
                masked_orig.append(orig_patches[i, mask_indices - cls_offset])
        
        # Compute reconstruction loss
        if len(masked_pred) > 0:
            masked_pred = torch.cat(masked_pred)
            masked_orig = torch.cat(masked_orig)
            
            # Compute MSE loss
            loss = F.mse_loss(masked_pred, masked_orig)
            
            # Compute per-sample errors
            per_sample_error = torch.zeros(B, device=feats.device)
            start_idx = 0
            
            for i in range(B):
                mask_count = mask[i, cls_offset:].sum().item()
                if mask_count > 0:
                    end_idx = start_idx + mask_count
                    sample_pred = masked_pred[start_idx:end_idx]
                    sample_orig = masked_orig[start_idx:end_idx]
                    per_sample_error[i] = F.mse_loss(sample_pred, sample_orig)
                    start_idx = end_idx
        else:
            # Fallback for edge case with no masked tokens
            loss = torch.tensor(0.0, device=feats.device)
            per_sample_error = torch.zeros(B, device=feats.device)
        
        return loss, per_sample_error
    
    def _compute_jepa_reconstruction(self, feats, context_mask, target_mask):
        """Compute JEPA-style prediction and error.
        
        Args:
            feats: features from the backbone
            context_mask: boolean mask indicating which patches are context (True = context)
            target_mask: boolean mask indicating which patches are targets (True = target)
            
        Returns:
            Tuple of (prediction loss, per-sample prediction errors)
        """
        B = feats.shape[0]
        
        # Get context and target tokens
        context_tokens = feats[context_mask].reshape(B, -1, feats.shape[-1])
        target_tokens = feats[target_mask].reshape(B, -1, feats.shape[-1])
        
        # Predict target tokens from context tokens
        # For simplicity, we'll use mean pooling of context tokens as input to the predictor
        context_pooled = context_tokens.mean(dim=1)
        pred_target = self.predictor(context_pooled).unsqueeze(1).expand_as(target_tokens)
        
        # Compute prediction loss
        loss = F.mse_loss(pred_target, target_tokens)
        
        # Compute per-sample errors
        per_sample_error = torch.mean((pred_target - target_tokens)**2, dim=(1, 2))
        
        return loss, per_sample_error
    
    
    
    def _compute_sample_weights_binary(self, per_sample_error, current_epoch):
        """Binary masking approach - completely blocks hard samples"""
        # Apply warmup to gradually include more samples
        if self.curriculum_warmup_epochs > 0:
            # Start with a low percentile (e.g., 10%) and gradually increase
            starting_percentile = 0.1
            current_percentile = starting_percentile + (1.0 - starting_percentile) * min(1.0, current_epoch / self.curriculum_warmup_epochs)
        else:
            current_percentile = 1.0  # Include all samples
        
        # Sort errors and find threshold
        sorted_errors, _ = torch.sort(per_sample_error)
        threshold_idx = int(current_percentile * len(sorted_errors))
        
        if threshold_idx < len(sorted_errors):
            threshold = sorted_errors[threshold_idx]
            threshold_value = threshold.item()  # Convert tensor to float safely
        else:
            threshold = float('inf')
            threshold_value = float('inf')  # Already a Python float
        
        # Create binary mask: 1 for samples below threshold, 0 for others
        mask = (per_sample_error <= threshold).float()
        
        # Ensure at least one sample is included
        if mask.sum() == 0:
            mask[per_sample_error.argmin()] = 1.0
        
        # Track curriculum metrics
        samples_included = mask.sum().item()
        batch_size = mask.size(0)
        percent_included = (samples_included / batch_size) * 100
        
        # Log detailed metrics
        self.log("curriculum_samples_included", samples_included, on_step=True, on_epoch=True)
        self.log("curriculum_percent_included", percent_included, on_step=True, on_epoch=True)
        self.log("curriculum_target_percentile", current_percentile * 100, on_step=True, on_epoch=True)
        self.log("curriculum_threshold_value", threshold_value, on_step=True, on_epoch=True)
        
        return mask
    
    def _compute_sample_weights_exponential(self, per_sample_error, current_epoch):
        """Exponential weighting - sharply diminishes influence of harder samples"""
        # Normalize errors to [0, 1]
        norm_errors = per_sample_error / (per_sample_error.max() + 1e-8)
        
        # Alpha controls sharpness of exponential weighting
        # Start with high alpha (steep dropoff) and gradually decrease
        max_alpha = 10.0  # Very steep initially
        min_alpha = 1.0   # Gentler at the end
        
        if self.curriculum_warmup_epochs > 0:
            # Decrease alpha over time (makes the weighting gentler)
            alpha = max_alpha - (max_alpha - min_alpha) * min(1.0, current_epoch / self.curriculum_warmup_epochs)
        else:
            alpha = min_alpha
        
        # Compute weights: exp(-α * error)
        weights = torch.exp(-alpha * norm_errors)
        
        # Normalize weights to mean=1
        weights = weights / (weights.mean() + 1e-8)
        
        self.log("curriculum_alpha", alpha, on_step=True, on_epoch=True)
        
        return weights
    
    def _compute_sample_weights_bands(self, per_sample_error, current_epoch):
        """Difficulty bands approach - focuses on progressively harder samples"""
        # Sort samples by difficulty
        B = per_sample_error.size(0)
        sorted_indices = torch.argsort(per_sample_error)
        
        # Define band size (e.g., 25% of samples)
        band_size = max(1, int(B * 0.25))
        
        # Calculate which band to focus on based on training progress
        if self.curriculum_warmup_epochs > 0:
            progress = min(1.0, current_epoch / self.curriculum_warmup_epochs)
        else:
            progress = 1.0
        
        # As training progresses, move from easiest to hardest band
        # At the end of warmup, all samples are weighted equally
        if progress >= 1.0:
            # After warmup, all samples have equal weight
            weights = torch.ones_like(per_sample_error)
        else:
            # Calculate which band to focus on
            band_start = min(B - band_size, int(progress * B))
            
            # Create weights
            weights = torch.zeros_like(per_sample_error)
            band_indices = sorted_indices[band_start:band_start + band_size]
            weights[band_indices] = 1.0
            
            # Optionally add some weight to other samples
            other_weight = 0.1
            weights[weights == 0] = other_weight
        
        # Normalize weights
        weights = weights / (weights.mean() + 1e-8)
        
        # Log which band we're focusing on
        band_start_pct = band_start / B if progress < 1.0 else 1.0
        self.log("curriculum_band_start", band_start_pct, on_step=True, on_epoch=True)
        
        return weights
    
    def _compute_sample_weights(self, per_sample_error, current_epoch):
        """Compute sample weights based on reconstruction errors using the selected strategy.
        
        Args:
            per_sample_error: per-sample reconstruction errors
            current_epoch: current epoch number for curriculum adjustment
            
        Returns:
            Tensor of sample weights for contrastive loss
        """
        if self.curriculum_strategy == "binary":
            return self._compute_sample_weights_binary(per_sample_error, current_epoch)
        elif self.curriculum_strategy == "exponential":
            return self._compute_sample_weights_exponential(per_sample_error, current_epoch)
        elif self.curriculum_strategy == "bands":
            return self._compute_sample_weights_bands(per_sample_error, current_epoch)
        else:
            # Default to exponential weighting
            return self._compute_sample_weights_exponential(per_sample_error, current_epoch)
    
    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass for the curriculum MoCo v3.
        
        This ensures the output preserves format compatibility with original MoCo v3.
        """
        out = super().forward(X)
        return out
    
    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Curriculum MoCo V3.
        
        Args:
            batch: a batch of data in the format of [img_indexes, [X], Y]
            batch_idx: index of the batch
            
        Returns:
            total loss composed of MoCo V3 contrastive loss and reconstruction loss
        """
        # Get the current epoch
        current_epoch = self.trainer.current_epoch
        
        # Extract indexes and images
        indexes = batch[0]
        X = batch[1]
        
        # Process both views separately
        out1 = super().forward(X[0])
        out2 = super().forward(X[1])
        
        momentum_out1 = super().momentum_forward(X[0])
        momentum_out2 = super().momentum_forward(X[1])
        
        # Get contrastive learning features
        q1 = out1["q"]  # Features from first view
        q2 = out2["q"]  # Features from second view
        
        k1 = momentum_out1["k"]  # Momentum features from first view
        k2 = momentum_out2["k"]  # Momentum features from second view
        
        # Forward features for reconstruction
        if self.is_vit:
            # For ViT, we need to work with the token sequence
            vit_feats = out1["feats_vit"]  # Use features from the first view
            
            if self.curriculum_type == "mae":
                # Create a random mask (True = masked)
                mask_ratio = self.masking_ratio
                num_tokens = vit_feats.shape[1]
                
                # Exclude CLS token from masking if present
                mask_start_idx = 1 if hasattr(self.backbone, "cls_token") else 0
                
                # Create random mask
                mask = torch.zeros(vit_feats.shape[0], num_tokens, dtype=torch.bool, device=vit_feats.device)
                mask_length = int(mask_ratio * (num_tokens - mask_start_idx))
                for i in range(vit_feats.shape[0]):
                    mask_indices = torch.randperm(num_tokens - mask_start_idx, device=vit_feats.device)[:mask_length]
                    mask[i, mask_indices + mask_start_idx] = True
                
                # Compute reconstruction loss and per-sample errors
                recon_loss, per_sample_error = self._compute_mae_reconstruction(vit_feats, mask, batch)
                
            else:  # JEPA-style
                # Split tokens into context and target sets
                num_tokens = vit_feats.shape[1]
                
                # Exclude CLS token from splitting if present
                split_start_idx = 1 if hasattr(self.backbone, "cls_token") else 0
                
                # Create masks for context and target (non-overlapping)
                target_ratio = self.masking_ratio
                target_length = int(target_ratio * (num_tokens - split_start_idx))
                
                context_mask = torch.ones(vit_feats.shape[0], num_tokens, dtype=torch.bool, device=vit_feats.device)
                target_mask = torch.zeros(vit_feats.shape[0], num_tokens, dtype=torch.bool, device=vit_feats.device)
                
                for i in range(vit_feats.shape[0]):
                    # Random permutation for each sample
                    indices = torch.randperm(num_tokens - split_start_idx, device=vit_feats.device)
                    target_indices = indices[:target_length] + split_start_idx
                    context_mask[i, target_indices] = False
                    target_mask[i, target_indices] = True
                
                # Compute prediction loss and per-sample errors
                recon_loss, per_sample_error = self._compute_jepa_reconstruction(vit_feats, context_mask, target_mask)
                
        else:
            # For ResNet/CNN backbones, compute reconstruction directly
            # Get original images - using first view
            orig_images = batch[1][0]  # Shape: [B, 3, 224, 224]
            B = out1["feats"].shape[0]
            
            # Decode features to image space (full 224x224 resolution)
            pred_images = self.decoder(out1["feats"])  # Shape: [B, 3, 224, 224]
            
            # Compute per-sample reconstruction errors (MSE for each sample)
            per_sample_error = ((pred_images - orig_images) ** 2).mean(dim=(1, 2, 3))
            
            # Compute overall reconstruction loss
            recon_loss = per_sample_error.mean()
        
        # Compute sample weights based on reconstruction errors
        sample_weights = self._compute_sample_weights(per_sample_error, current_epoch)
        
        # Compute the weighted contrastive loss
        # First direction: q1 to k2
        q1_norm = F.normalize(q1, dim=-1)
        k2_norm = F.normalize(k2, dim=-1)
        
        # Gather all targets from all GPUs
        k2_all = gather(k2_norm)
        
        # Compute logits
        logits1 = torch.einsum("nc,mc->nm", [q1_norm, k2_all]) / self.temperature
        
        # Create labels - we want to match the i-th query with the i-th key
        batch_size = q1_norm.shape[0]
        labels = torch.arange(batch_size, device=q1_norm.device) + batch_size * self.trainer.global_rank
        
        # Compute per-sample cross-entropy loss
        per_sample_nll1 = F.cross_entropy(logits1, labels, reduction='none')
        
        # Apply sample weights
        weighted_contrastive_loss1 = (per_sample_nll1 * sample_weights).mean()
        
        # Second direction: q2 to k1
        q2_norm = F.normalize(q2, dim=-1)
        k1_norm = F.normalize(k1, dim=-1)
        k1_all = gather(k1_norm)
        logits2 = torch.einsum("nc,mc->nm", [q2_norm, k1_all]) / self.temperature
        per_sample_nll2 = F.cross_entropy(logits2, labels, reduction='none')
        weighted_contrastive_loss2 = (per_sample_nll2 * sample_weights).mean()
        
        # Combine both directions
        weighted_contrastive_loss = (weighted_contrastive_loss1 + weighted_contrastive_loss2) / 2
        
        # Total loss is weighted sum of contrastive and reconstruction losses
        total_loss = weighted_contrastive_loss + self.curriculum_weight * recon_loss
        
        # Log metrics
        metrics = {
            "train_contrastive_loss": weighted_contrastive_loss,
            "train_reconstruction_loss": recon_loss,
            "train_total_loss": total_loss,
            "train_sample_weights_mean": sample_weights.mean(),
            "train_sample_weights_std": sample_weights.std(),
            "train_reconstruction_error_mean": per_sample_error.mean(),
        }
        self.log_dict(metrics, on_epoch=True, on_step=True, sync_dist=True)
        
        return total_loss