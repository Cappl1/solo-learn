from typing import Any, Dict, List, Sequence, Tuple

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.mocov3 import mocov3_loss_func
from solo.methods.mocov3 import MoCoV3
from solo.utils.misc import gather
from solo.utils.lars import LARS  

class SelectiveGradientFunction(torch.autograd.Function):
    """Custom autograd function that selectively scales gradients based on sample weights."""
    
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(weights)
        return inputs.clone()  # Clone to ensure we don't modify the original tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors
        # Scale the gradients by weights
        # Expand weights to match grad_output shape if needed
        if weights.dim() == 1 and grad_output.dim() > 1:
            # For batch-wise weights
            scaled_grads = grad_output * weights.view(-1, *([1] * (grad_output.dim() - 1)))
        else:
            scaled_grads = grad_output * weights
        return scaled_grads, None  # Return None for weights gradient


def apply_sample_weights(tensor, weights):
    """Apply sample weights to gradients during backpropagation."""
    return SelectiveGradientFunction.apply(tensor, weights)


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
                curriculum_reverse (bool): if True, prioritize hard samples first instead of easy ones
        """
        super().__init__(cfg)
        
        # Get curriculum learning parameters
        self.curriculum_type = cfg.method_kwargs.get("curriculum_type", "mae")  # "mae" or "jepa"
        self.curriculum_strategy = cfg.method_kwargs.get("curriculum_strategy", "exponential")  # "binary", "exponential", or "bands"
        self.curriculum_warmup_epochs = cfg.method_kwargs.get("curriculum_warmup_epochs", 10)
        self.curriculum_weight = cfg.method_kwargs.get("curriculum_weight", 1.0)
        self.masking_ratio = cfg.method_kwargs.get("reconstruction_masking_ratio", 0.75)
        self.curriculum_reverse = cfg.method_kwargs.get("curriculum_reverse", False)  # New: Reverse curriculum direction
        
        # Store optimizer name to use in configure_optimizers
        self.optimizer = cfg.optimizer.name if hasattr(cfg.optimizer, 'name') else 'sgd'
        
        # Get optimizer settings
        if hasattr(cfg.optimizer, 'lr'):
            self.learning_rate = cfg.optimizer.lr
        if hasattr(cfg.optimizer, 'weight_decay'):
            self.weight_decay = cfg.optimizer.weight_decay
        if hasattr(cfg.optimizer, 'momentum'):
            self.momentum = cfg.optimizer.momentum
        else:
            self.momentum = 0.9  # Default momentum
        
        # Store LARS specific settings if they exist
        self.lars = True if self.optimizer.lower() == 'lars' else False
        
        # Scheduler settings
        self.scheduler = cfg.scheduler.name if hasattr(cfg, 'scheduler') and hasattr(cfg.scheduler, 'name') else None
        if self.scheduler == "warmup_cosine":
            self.warmup_epochs = cfg.scheduler.warmup_epochs
            self.max_epochs = cfg.max_epochs
            if hasattr(cfg.scheduler, 'warmup_start_lr'):
                self.warmup_start_lr = cfg.scheduler.warmup_start_lr
            if hasattr(cfg.scheduler, 'min_lr'):
                self.min_lr = cfg.scheduler.min_lr
        elif self.scheduler == "step":
            if hasattr(cfg.scheduler, 'step_size'):
                self.lr_step_size = cfg.scheduler.step_size
            if hasattr(cfg.scheduler, 'gamma'):
                self.lr_gamma = cfg.scheduler.gamma
        
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
                self.predictor_jepa = self._build_jepa_predictor(embed_dim)
        else:
            # For ResNet/CNN backbones
            # Get the feature dimension for ResNet
            self.features_dim = getattr(self.backbone, 'inplanes', 512)
            
            if self.curriculum_type == "mae":
                # For MAE-style, use our proper decoder for 224x224 images
                self.decoder = self._build_resnet_decoder()
            else:  # JEPA style
                # For JEPA-style, create a predictor for feature prediction
                print("Initializing JEPA predictor for ResNet")
                self.predictor_jepa = self._build_resnet_predictor()

    def configure_ddp(self, ddp_kwargs):
        """Configure the DDP strategy."""
        # Add find_unused_parameters=True to the DDP settings
        ddp_kwargs["find_unused_parameters"] = True
        print("DDP configured with find_unused_parameters=True")
        return ddp_kwargs 
        

    
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
    
    def _build_resnet_predictor(self) -> nn.Module:
        """Build a predictor for ResNet features."""
        predictor_dim = self.features_dim
        predictor = nn.Sequential(
            nn.Linear(predictor_dim, predictor_dim * 2),
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
            
        if "curriculum_reverse" not in cfg.method_kwargs:
            cfg.method_kwargs.curriculum_reverse = False
        
        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Returns learnable parameters for the optimizer.
        
        We need to be careful to avoid parameter duplication between parent and child classes.
        """
        # Get base params from the parent class (backbone, projector, predictor)
        base_params = super().learnable_params
        
        # Add our curriculum-specific parameters as a separate group
        curriculum_params = []
        
        if self.curriculum_type == "mae":
            curriculum_params.append({"name": "decoder", "params": self.decoder.parameters()})
        else:  # JEPA
            curriculum_params.append({"name": "predictor_jepa", "params": self.predictor_jepa.parameters()})
        
        return base_params + curriculum_params
    
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
        pred_target = self.predictor_jepa(context_pooled).unsqueeze(1).expand_as(target_tokens)
        
        # Compute prediction loss
        loss = F.mse_loss(pred_target, target_tokens)
        
        # Compute per-sample errors with added noise to ensure variation
        per_sample_error = torch.mean((pred_target - target_tokens)**2, dim=(1, 2))
        
        # Add small random noise to ensure variation in errors (helps with curriculum)
        noise_scale = per_sample_error.mean() * 0.01  # Scale noise to be 1% of mean error
        noise = torch.rand_like(per_sample_error) * noise_scale
        per_sample_error = per_sample_error + noise
        
        return loss, per_sample_error
    
    def _compute_sample_weights_binary(self, per_sample_error, current_epoch):
        """Binary masking approach - completely blocks hard/easy samples.
        
        If curriculum_reverse is True, prioritize hard samples first.
        """
        # Apply warmup to gradually include more samples
        if self.curriculum_warmup_epochs > 0:
            # Start with a low percentile (e.g., 10%) and gradually increase
            starting_percentile = 0.1
            current_percentile = starting_percentile + (1.0 - starting_percentile) * min(1.0, current_epoch / self.curriculum_warmup_epochs)
        else:
            current_percentile = 1.0  # Include all samples
        
        # After curriculum period, include all samples
        if current_percentile >= 1.0:
            return torch.ones_like(per_sample_error)
            
        # Sort errors and find threshold
        sorted_errors, sorted_indices = torch.sort(per_sample_error)
        
        # If all errors are identical, add small random noise to create variation
        if per_sample_error.std() < 1e-6:
            noise = torch.rand_like(per_sample_error) * 0.01
            per_sample_error = per_sample_error + noise
            sorted_errors, sorted_indices = torch.sort(per_sample_error)
        
        # If reversed, we want the hardest samples first (highest errors)
        if self.curriculum_reverse:
            # Invert the indices to get hardest first
            sorted_indices = sorted_indices.flip(0)
            sorted_errors = per_sample_error[sorted_indices]
        
        threshold_idx = int(current_percentile * len(sorted_errors))
        
        if threshold_idx < len(sorted_errors):
            threshold = sorted_errors[threshold_idx]
        else:
            threshold = float('inf')
        
        # Create binary mask based on threshold
        if self.curriculum_reverse:
            # For reverse curriculum, include samples with errors >= threshold
            mask = (per_sample_error >= threshold).float()
        else:
            # For normal curriculum, include samples with errors <= threshold
            mask = (per_sample_error <= threshold).float()
        
        # Ensure at least one sample is included
        if mask.sum() == 0:
            if self.curriculum_reverse:
                # Include the hardest sample
                mask[per_sample_error.argmax()] = 1.0
            else:
                # Include the easiest sample
                mask[per_sample_error.argmin()] = 1.0
        
        # Track curriculum metrics
        samples_included = mask.sum().item()
        batch_size = mask.size(0)
        percent_included = (samples_included / batch_size) * 100
        
        # Enhanced logging for binary strategy
        self.log("curriculum_direction", 1.0 if self.curriculum_reverse else 0.0, on_step=False, on_epoch=True)
        self.log("curriculum_samples_included", samples_included, on_step=True, on_epoch=True)
        self.log("curriculum_percent_included", percent_included, on_step=True, on_epoch=True)
        self.log("curriculum_target_percentile", current_percentile * 100, on_step=True, on_epoch=True)
        
        # Log error statistics
        self.log("curriculum_errors_min", per_sample_error.min(), on_step=False, on_epoch=True)
        self.log("curriculum_errors_max", per_sample_error.max(), on_step=False, on_epoch=True)
        self.log("curriculum_errors_median", per_sample_error.median(), on_step=False, on_epoch=True)
        
        return mask
    
    def _compute_sample_weights_exponential(self, per_sample_error, current_epoch):
        """Exponential weighting - controls influence of samples by difficulty.
        
        If curriculum_reverse is True, prioritize hard samples (high weights for high errors).
        """
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
        
        # Compute weights based on normalized errors
        if self.curriculum_reverse:
            # For reversed curriculum: exp(α * error) - higher weight to harder samples
            weights = torch.exp(alpha * norm_errors)
        else:
            # For normal curriculum: exp(-α * error) - higher weight to easier samples
            weights = torch.exp(-alpha * norm_errors)
        
        # Normalize weights to mean=1
        weights = weights / (weights.mean() + 1e-8)
        
        # Enhanced logging for exponential weighting
        curriculum_direction = "hard-to-easy" if self.curriculum_reverse else "easy-to-hard"
        self.log("curriculum_direction", 1.0 if self.curriculum_reverse else 0.0, on_step=False, on_epoch=True)
        self.log("curriculum_alpha", alpha, on_step=True, on_epoch=True)
        
        # Log distribution statistics
        self.log("curriculum_weights_min", weights.min(), on_step=False, on_epoch=True)
        self.log("curriculum_weights_max", weights.max(), on_step=False, on_epoch=True)
        self.log("curriculum_weights_median", weights.median(), on_step=False, on_epoch=True)
        
        self.log("curriculum_errors_min", per_sample_error.min(), on_step=False, on_epoch=True)
        self.log("curriculum_errors_max", per_sample_error.max(), on_step=False, on_epoch=True)
        self.log("curriculum_errors_median", per_sample_error.median(), on_step=False, on_epoch=True)
        
        # Calculate effective sample size (sum of weights / max weight)
        # This tells us how many samples are effectively contributing
        effective_samples = weights.sum() / (weights.max() + 1e-8)
        self.log("curriculum_effective_samples", effective_samples, on_step=True, on_epoch=True)
        
        # Calculate percentage of samples with significant weight (>0.5 of max weight)
        significant_weight_threshold = weights.max() * 0.5
        significant_samples_percent = (weights > significant_weight_threshold).float().mean() * 100
        self.log("curriculum_significant_samples_percent", significant_samples_percent, on_step=True, on_epoch=True)
        
        return weights
    
    def _compute_sample_weights_bands(self, per_sample_error, current_epoch):
        """Difficulty bands approach - focuses on specific difficulty bands.
        
        If curriculum_reverse is True, start with hardest band and move to easier bands.
        """
        # Sort samples by difficulty
        B = per_sample_error.size(0)
        sorted_indices = torch.argsort(per_sample_error)
        
        # If reversed, flip the order to get hardest first
        if self.curriculum_reverse:
            sorted_indices = sorted_indices.flip(0)
        
        # Define band size (e.g., 25% of samples)
        band_size = max(1, int(B * 0.25))
        
        # Calculate which band to focus on based on training progress
        if self.curriculum_warmup_epochs > 0:
            progress = min(1.0, current_epoch / self.curriculum_warmup_epochs)
        else:
            progress = 1.0
        
        # As training progresses, move between bands
        # At the end of warmup, all samples are weighted equally
        if progress >= 1.0:
            # After warmup, all samples have equal weight
            weights = torch.ones_like(per_sample_error)
            band_start = B  # For logging, indicate we're past bands
        else:
            # Calculate which band to focus on
            band_start = min(B - band_size, int(progress * B))
            band_end = band_start + band_size
            
            # Create weights
            weights = torch.zeros_like(per_sample_error)
            band_indices = sorted_indices[band_start:band_end]
            weights[band_indices] = 1.0
            
            # Optionally add some weight to other samples
            other_weight = 0.1
            weights[weights == 0] = other_weight
        
        # Normalize weights
        weights = weights / (weights.mean() + 1e-8)
        
        # Enhanced logging for bands strategy
        curriculum_direction = "hard-to-easy" if self.curriculum_reverse else "easy-to-hard"
        self.log("curriculum_direction", 1.0 if self.curriculum_reverse else 0.0, on_step=False, on_epoch=True)
        band_start_pct = band_start / B if progress < 1.0 else 1.0
        self.log("curriculum_band_start", band_start_pct, on_step=True, on_epoch=True)
        self.log("curriculum_progress", progress, on_step=True, on_epoch=True)
        
        # Log which difficulty band we're focusing on
        if progress < 1.0:
            band_percentile_start = (band_start / B) * 100
            band_percentile_end = min(100, ((band_start + band_size) / B) * 100)
            band_description = f"{band_percentile_start:.1f}%-{band_percentile_end:.1f}%"
        else:
            band_description = "All samples"
            
        # We can't log strings directly, so we'll log numerically
        self.log("curriculum_band_percentile_start", band_start / B * 100, on_step=False, on_epoch=True)
        self.log("curriculum_band_percentile_end", min(100, (band_start + band_size) / B * 100), on_step=False, on_epoch=True)
        
        # Log error statistics within the current band
        if progress < 1.0:
            band_errors = per_sample_error[band_indices]
            self.log("curriculum_band_error_min", band_errors.min(), on_step=False, on_epoch=True)
            self.log("curriculum_band_error_max", band_errors.max(), on_step=False, on_epoch=True)
            self.log("curriculum_band_error_mean", band_errors.mean(), on_step=False, on_epoch=True)
        
        # Log overall error statistics
        self.log("curriculum_errors_min", per_sample_error.min(), on_step=False, on_epoch=True)
        self.log("curriculum_errors_max", per_sample_error.max(), on_step=False, on_epoch=True)
        self.log("curriculum_errors_median", per_sample_error.median(), on_step=False, on_epoch=True)
        
        # Calculate how many samples have significant weight (>0.5 of max weight)
        significant_weight_threshold = weights.max() * 0.5
        significant_samples = (weights > significant_weight_threshold).float().sum()
        self.log("curriculum_significant_samples", significant_samples, on_step=True, on_epoch=True)
        
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
        """Training step with automatic optimization through PyTorch Lightning.
        
        This simplifies the code by using PyTorch Lightning's automatic optimization
        instead of manual optimization. The momentum update is now handled by
        the parent class's on_train_batch_end hook.
        """
        # ---------------------------------------------- 1. Unpack batch
        indexes, views = batch[0], batch[1]
        x1, x2 = views[0], views[1]

        # ---------------------------------------------- 2. Forward passes
        out1 = super().forward(x1)  # online
        out2 = super().forward(x2)
        m_out1 = super().momentum_forward(x1)  # momentum
        m_out2 = super().momentum_forward(x2)

        q1, q2 = out1["q"], out2["q"]
        k1, k2 = m_out1["k"].detach(), m_out2["k"].detach()

        # ---------------------------------------------- 3. Reconstruction branch
        # Compute reconstruction and get per-sample errors
        if self.is_vit:
            feats = out1["feats_vit"].detach()  # detach = no backbone grad
            if self.curriculum_type == "mae":
                # Create a random mask (True = masked)
                mask_ratio = self.masking_ratio
                num_tokens = feats.shape[1]
                
                # Exclude CLS token from masking if present
                mask_start_idx = 1 if hasattr(self.backbone, "cls_token") else 0
                
                # Create random mask
                mask = torch.zeros(feats.shape[0], num_tokens, dtype=torch.bool, device=feats.device)
                mask_length = int(mask_ratio * (num_tokens - mask_start_idx))
                for i in range(feats.shape[0]):
                    mask_indices = torch.randperm(num_tokens - mask_start_idx, device=feats.device)[:mask_length]
                    mask[i, mask_indices + mask_start_idx] = True
                
                recon_loss, per_error = self._compute_mae_reconstruction(feats, mask, batch)
            else:  # JEPA
                # Split tokens into context and target sets
                num_tokens = feats.shape[1]
                
                # Exclude CLS token from splitting if present
                split_start_idx = 1 if hasattr(self.backbone, "cls_token") else 0
                
                # Create masks for context and target (non-overlapping)
                target_ratio = self.masking_ratio
                target_length = int(target_ratio * (num_tokens - split_start_idx))
                
                context_mask = torch.ones(feats.shape[0], num_tokens, dtype=torch.bool, device=feats.device)
                target_mask = torch.zeros(feats.shape[0], num_tokens, dtype=torch.bool, device=feats.device)
                
                for i in range(feats.shape[0]):
                    # Random permutation for each sample
                    indices = torch.randperm(num_tokens - split_start_idx, device=feats.device)
                    target_indices = indices[:target_length] + split_start_idx
                    context_mask[i, target_indices] = False
                    target_mask[i, target_indices] = True
                
                recon_loss, per_error = self._compute_jepa_reconstruction(feats, context_mask, target_mask)
        else:  # ResNet
            if self.curriculum_type == "mae":
                pred_img = self.decoder(out1["feats"].detach())
                tgt_img = x1
                per_error = ((pred_img - tgt_img) ** 2).mean(dim=(1, 2, 3))
                recon_loss = per_error.mean()
            else:  # JEPA ResNet
                pred_feats = self.predictor_jepa(out1["feats"].detach())
                per_error = F.mse_loss(pred_feats, out2["feats"].detach(),
                                    reduction='none').mean(dim=1)
                recon_loss = per_error.mean()

        # ---------------------------------------------- 4. Compute curriculum weights
        weights = self._compute_sample_weights(
            per_error, self.trainer.current_epoch
        )

        # ---------------------------------------------- 5. Compute weighted contrastive loss
        # Apply weights to the queries
        q1_weighted = apply_sample_weights(q1, weights)
        q2_weighted = apply_sample_weights(q2, weights)
        
        # Use the original loss function from solo.losses.mocov3
        contrastive_loss = mocov3_loss_func(
            q1_weighted, k2, temperature=self.temperature
        ) + mocov3_loss_func(
            q2_weighted, k1, temperature=self.temperature
        )

        # ---------------------------------------------- 6. Combine losses
        total_loss = contrastive_loss + recon_loss

        # ---------------------------------------------- 7. Logging
        self.log_dict(
            {
                "train_contrastive_loss": contrastive_loss,
                "train_reconstruction_loss": recon_loss,
                "train_total_loss": total_loss,
                "train_sample_weights_mean": weights.mean(),
                "train_sample_weights_std": weights.std(),
                "train_reconstruction_err_mean": per_error.mean(),
                "train_samples_included": (weights > 0).float().sum(),
                "train_percent_included": (weights > 0).float().mean() * 100,
                "curriculum_phase": self.trainer.current_epoch / max(1, self.curriculum_warmup_epochs),
            },
            on_step=True, on_epoch=True, sync_dist=True,
        )

        # Lightning will automatically handle optimization when we return the loss
        return total_loss
    
    def update_momentum_manually(self):
        """Updates momentum encoder via exponential moving average.
        
        Manually implemented since we can't rely on parent's update_momentum method.
        """
        # Get momentum value
        base_tau = self.momentum.base_tau
        current_tau = base_tau
        
        # Update each momentum pair
        for online, target in self.momentum_pairs:
            for online_param, target_param in zip(online.parameters(), target.parameters()):
                target_param.data = current_tau * target_param.data + (1 - current_tau) * online_param.data
    
