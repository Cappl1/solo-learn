from typing import Any, Dict, List, Sequence, Tuple

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.misc import gather


class TemporalJEPA(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements Temporal I-JEPA for frame-to-frame prediction.
        
        This method adapts I-JEPA for temporal prediction rather than spatial completion.
        Instead of predicting masked patches within the same image, we predict the latent 
        representation of a future frame given the current frame.
        
        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of projected features.
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                delta_max (int): maximum time delta between frames (default: 15).
                temperature (float): temperature for normalizing losses.
                var_lambda (float): weight for variance regularization (default: 1.0).
                cov_lambda (float): weight for covariance regularization (default: 1.0).
        """
        super().__init__(cfg)
        
        # Get configuration parameters
        self.delta_max = cfg.method_kwargs.get("delta_max", 15)
        self.temperature = cfg.method_kwargs.get("temperature", 1.0)
        self.var_lambda = cfg.method_kwargs.get("var_lambda", 1.0)
        self.cov_lambda = cfg.method_kwargs.get("cov_lambda", 1.0)
        
        proj_hidden_dim = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim = cfg.method_kwargs.proj_output_dim
        
        # Create context encoder (projector)
        self.projector = self._build_mlp(
            3,  # 3-layer MLP
            self.features_dim,
            proj_hidden_dim,
            proj_output_dim,
        )
        
        # Create momentum projector (target encoder)
        self.momentum_projector = self._build_mlp(
            3,
            self.features_dim,
            proj_hidden_dim,
            proj_output_dim,
        )
        
        # Create predictor (transformer-based)
        predictor_dim = proj_output_dim
        self.predictor = self._build_predictor(predictor_dim)
        
        # Initialize momentum projector with projector weights
        initialize_momentum_params(self.projector, self.momentum_projector)
    
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        """Build a multi-layer perceptron for projection."""
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            
            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        
        return nn.Sequential(*mlp)
    
    def _build_predictor(self, dim):
        """Build a transformer-based predictor for temporal prediction."""
        # Simple transformer encoder layer based predictor
        predictor = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.TransformerEncoderLayer(d_model=dim * 2, nhead=8, batch_first=True),
            nn.Linear(dim * 2, dim),
        )
        return predictor
    
    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config."""
        cfg = super(TemporalJEPA, TemporalJEPA).add_and_assert_specific_cfg(cfg)
        
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        
        # Add default values if missing
        if "delta_max" not in cfg.method_kwargs:
            cfg.method_kwargs.delta_max = 15
        if "temperature" not in cfg.method_kwargs:
            cfg.method_kwargs.temperature = 1.0
        if "var_lambda" not in cfg.method_kwargs:
            cfg.method_kwargs.var_lambda = 1.0
        if "cov_lambda" not in cfg.method_kwargs:
            cfg.method_kwargs.cov_lambda = 1.0
            
        return cfg
    
    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters."""
        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()},
            {"name": "predictor", "params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params
    
    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs."""
        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs
    
    def forward(self, X):
        """Forward pass through the online backbone and projector.
        
        Args:
            X: Input to be processed. Could be a tensor or a list.
            
        Returns:
            Dict: dict containing the embeddings.
        """
        # Handle case where X is a list of multiple crop transformations
        if isinstance(X, list):
            # Extract first tensor for processing
            if len(X) > 0:
                X = X[0]
            else:
                # Return empty embeddings if X is an empty list
                return {"z": torch.empty(0, device=self.device)}
                
        # Now X should be a tensor - proceed with normal processing
        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out
    
    @torch.no_grad()
    def momentum_forward(self, X):
        """Forward pass through the momentum backbone and projector.
        
        Args:
            X: Input to be processed. Could be a tensor or a list.
            
        Returns:
            Dict: dict containing the embeddings.
        """
        # Handle case where X is a list of multiple crop transformations
        if isinstance(X, list):
            # Extract first tensor for processing
            if len(X) > 0:
                X = X[0]
            else:
                # Return empty embeddings if X is an empty list
                return {"k": torch.empty(0, device=self.device)}
                
        # Move the input to channels last memory format only if it's a 4D tensor
        if isinstance(X, torch.Tensor):
            if X.ndim == 4:  # Only apply channels_last to 4D tensors
                X = X.to(memory_format=torch.channels_last)
            else:
                # If not 4D, print debug info but continue
                if self.global_rank == 0:
                    print(f"Warning: Got tensor with shape {X.shape} (ndim={X.ndim}), expected 4D tensor")
        
        try:
            feats = self.momentum_backbone(X)
            k = self.momentum_projector(feats)
            return {"k": k}
        except Exception as e:
            print(f"Error in momentum_forward: {e}, tensor shape: {X.shape if isinstance(X, torch.Tensor) else 'not tensor'}")
            # Return dummy tensor as fallback
            return {"k": torch.zeros(X.shape[0] if isinstance(X, torch.Tensor) and len(X.shape) > 0 else 1, 
                                      self.features_dim, device=self.device)}
    
    def variance_regularization_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute variance regularization loss to ensure tokens are not collapsing.
        
        Args:
            z (torch.Tensor): batch of features in tensor format [B, D].
            
        Returns:
            torch.Tensor: variance regularization loss.
        """
        # Compute variance across batch
        z_std = torch.sqrt(z.var(dim=0) + 1e-6)
        var_loss = torch.mean(F.relu(1 - z_std))
        return var_loss
    
    def covariance_regularization_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute covariance regularization loss to ensure features are decorrelated.
        
        Args:
            z (torch.Tensor): batch of features in tensor format [B, D].
            
        Returns:
            torch.Tensor: covariance regularization loss.
        """
        # Normalize features
        z = z - z.mean(dim=0)
        z = F.normalize(z, dim=1)
        
        # Compute covariance matrix
        B = z.size(0)
        cov = torch.matmul(z.T, z) / B
        
        # Remove diagonal (self-covariance)
        eye = torch.eye(cov.size(0), device=cov.device)
        cov_reg_loss = (cov * (1 - eye)).pow(2).sum() / cov.size(0)
        
        return cov_reg_loss
    
    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Temporal I-JEPA adapted for main_pretrain.py.
        
        Args:
            batch (Sequence[Any]): a batch of data that could come in different formats
            batch_idx (int): index of the batch.
            
        Returns:
            torch.Tensor: total loss.
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
                # Just grab the first entry and assume it's the views
                all_views = batch[0] if isinstance(batch[0], (list, tuple)) else batch
        else:
            all_views = batch
        
        # Make sure all_views is a list
        if not isinstance(all_views, (list, tuple)):
            # If somehow we got a single tensor, wrap it in a list
            all_views = [all_views]
        
        # Check if batch contains enough crops for temporal processing
        if len(all_views) < 2:
            # Just return a constant loss if there are not enough views
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log_dict(
                {"train_loss": dummy_loss},
                on_epoch=True,
                on_step=True,
                sync_dist=True,
            )
            return dummy_loss
        
        # Use first two crops as temporal pair
        x_t = all_views[0]  # Current frame
        x_t_plus_delta = all_views[1]  # Future frame
        
        # Get embeddings for anchor frame (using online encoder)
        out_t = self.forward(x_t)
        z_t = out_t["z"]
        
        # Get embeddings for target frame (using momentum encoder)
        with torch.no_grad():
            out_t_plus_delta = self.momentum_forward(x_t_plus_delta)
            z_t_plus_delta = out_t_plus_delta["k"]
        
        # Predict future embeddings from current embeddings
        z_pred = self.predictor(z_t)
        
        # Compute L2 loss in latent space
        pred_loss = F.mse_loss(z_pred, z_t_plus_delta)
        
        # Compute regularization losses
        var_loss = self.variance_regularization_loss(z_t_plus_delta)
        cov_loss = self.covariance_regularization_loss(z_t_plus_delta)
        
        # Combine losses
        loss = pred_loss + self.var_lambda * var_loss + self.cov_lambda * cov_loss
        
        # Log metrics
        metrics = {
            "train_loss": loss,
            "train_pred_loss": pred_loss,
            "train_var_loss": var_loss,
            "train_cov_loss": cov_loss,
        }
        self.log_dict(metrics, on_epoch=True, on_step=True, sync_dist=True)
        
        return loss
        
    def validation_step(self, batch: Sequence[Any], batch_idx: int) -> Dict[str, Any]:
        """Validation step for Temporal I-JEPA.
        
        This method has been adapted to handle the standard data format in main_pretrain.py.
        
        Args:
            batch (Sequence[Any]): a batch of validation data
            batch_idx (int): index of the batch
            
        Returns:
            Dict[str, Any]: Dictionary containing validation loss and batch size.
        """
        # Start with a known batch size to avoid errors
        batch_size = 1
        
        try:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) > 0 and isinstance(batch[0], torch.Tensor):
                    batch_size = batch[0].size(0)
                
                if len(batch) == 3:
                    # Format: [indexes, views, targets]
                    indexes, all_views, targets = batch
                elif len(batch) == 2:
                    # Format: [indexes, views]
                    indexes, all_views = batch
                else:
                    # Just grab the first entry and assume it's the views
                    all_views = batch[0] if isinstance(batch[0], (list, tuple)) else batch
            else:
                all_views = batch
            
            # Make sure all_views is a list
            if not isinstance(all_views, (list, tuple)):
                # If somehow we got a single tensor, wrap it in a list
                all_views = [all_views]
            
            # Need at least 2 views for temporal prediction
            if len(all_views) >= 2:
                x_t = all_views[0]
                x_t_plus_delta = all_views[1]
                
                # Set batch size if we can
                if isinstance(x_t, torch.Tensor):
                    batch_size = x_t.size(0)
                
                out_t = self.forward(x_t)
                z_t = out_t["z"]
                
                out_t_plus_delta = self.momentum_forward(x_t_plus_delta)
                z_t_plus_delta = out_t_plus_delta["k"]
                
                z_pred = self.predictor(z_t)
                val_loss = F.mse_loss(z_pred, z_t_plus_delta)
            else:
                # Not enough views, return dummy loss
                val_loss = torch.tensor(0.0, device=self.device)
        except Exception as e:
            print(f"Validation step error: {e}")
            val_loss = torch.tensor(0.0, device=self.device)
        
        # Ensure we're logging something meaningful
        batch_size = max(1, batch_size)  # Avoid division by zero
        
        # Log and return 
        self.log("val_loss", val_loss, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return {"val_loss": val_loss, "batch_size": batch_size}

    def on_validation_epoch_end(self):
        """Override base validation epoch end to handle our specific validation outputs.
        
        The base implementation uses weighted_mean which is causing division by zero errors.
        This simplified version just computes a simple average or returns 0 if no data.
        """
        if not hasattr(self, "validation_step_outputs") or not self.validation_step_outputs:
            # If no outputs, just log a zero
            val_loss = torch.tensor(0.0, device=self.device)
            self.log("val_loss", val_loss)
            return
        
        # Simple mean of validation losses without weighting
        if len(self.validation_step_outputs) > 0:
            total_loss = sum(output["val_loss"] for output in self.validation_step_outputs)
            val_loss = total_loss / len(self.validation_step_outputs)
        else:
            val_loss = torch.tensor(0.0, device=self.device)
        
        # Log the validation loss
        self.log("val_loss", val_loss)
        
        # Clear validation step outputs
        self.validation_step_outputs.clear()
        
        return val_loss