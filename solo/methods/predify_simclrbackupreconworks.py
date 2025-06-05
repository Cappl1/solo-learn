import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Sequence, Tuple
import omegaconf

from solo.methods.base import BaseMethod
from solo.utils.pcoder import PCoder
from solo.losses.simclr import simclr_loss_func


class PredifySimCLR(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements PredifySimCLR: Predictive Coding + SimCLR with temporal support.
        
        This method combines hierarchical predictive coding with SimCLR contrastive learning.
        Can work with:
        1. Standard augmented views (like original SimCLR)
        2. Temporal views (using time_window from dataset)
        3. No augmentations (using only temporal structure)

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of projected features.
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
                timesteps (int): number of predictive dynamics timesteps.
                pred_loss_weight (float): weight for predictive loss.
                ffm (list): feedforward multipliers for each PCoder.
                fbm (list): feedback multipliers for each PCoder.
                erm (list): error multipliers for each PCoder.
                use_local_updates (bool): whether to use local updates.
                use_temporal_pairs (bool): whether to use temporal neighbors as positive pairs.
                temporal_temperature (float): temperature for temporal contrastive loss.
                pcoder_grad_scale (float): scaling factor for PCoder gradients to backbone.
                enable_pcoder_grads (bool): whether to allow PCoder gradients to flow to backbone.
        """
        super().__init__(cfg)
        
        # Flag to control KNN feature type
        self.use_projected_features_for_knn = \
            "vgg" in self.backbone_name or \
            "efficientnet" in self.backbone_name
        
        # Predictive coding parameters
        self.timesteps: int = cfg.method_kwargs.timesteps
        self.pred_loss_weight: float = cfg.method_kwargs.pred_loss_weight
        self.use_local_updates: bool = cfg.method_kwargs.use_local_updates
        
        # Temporal parameters - simplified since dataset handles temporal pairing
        self.use_temporal_pairs: bool = cfg.method_kwargs.get("use_temporal_pairs", True)
        self.temporal_temperature: float = cfg.method_kwargs.get("temporal_temperature", 0.07)
        
        # PCoder gradient parameters
        self.pcoder_grad_scale: List[float] = cfg.method_kwargs.get("pcoder_grad_scale", [0.1, 0.08, 0.06, 0.04, 0.02])
        self.enable_pcoder_grads: bool = cfg.method_kwargs.get("enable_pcoder_grads", False)
        
        # PCoder multipliers
        self.ffm = cfg.method_kwargs.ffm
        self.fbm = cfg.method_kwargs.fbm
        self.erm = cfg.method_kwargs.erm
        
        # Standard SimCLR parameters
        self.temperature: float = cfg.method_kwargs.temperature
        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        
        # Only support VGG16 for now
        assert "vgg" in self.backbone_name, "PredifySimCLR currently only supports VGG16 backbone"
        
        # VGG16 layer positions and dimensions for PCoder hooks
        self.layer_positions = [3, 8, 15, 22, 29]  # ReLU after conv layers
        self.layer_dims = [64, 128, 256, 512, 512]
        
        # Create predictive coding modules
        self.create_pcoders()
        
        # Standard SimCLR projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        
        # Store intermediate representations
        self.representations = {}
        
        # Register hooks for feature extraction
        self.register_extraction_hooks()
    
    def create_pcoders(self):
        """Create PCoder modules for hierarchical prediction"""
        
        # PCoder 1: Layer 1 (64) predicts INPUT IMAGE (3)
        self.pcoder1 = PCoder(
            nn.Sequential(
                nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2),
                nn.Tanh()
            ),
            has_feedback=True,
            random_init=False
        )
        
        # PCoder 2: Layer 2 (128) predicts Layer 1 (64)
        self.pcoder2 = PCoder(
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=10, stride=2, padding=4),
                nn.ReLU(inplace=True)
            ),
            has_feedback=True,
            random_init=False
        )
        
        # PCoder 3: Layer 3 (256) predicts Layer 2 (128)
        self.pcoder3 = PCoder(
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=10, stride=2, padding=4),
                nn.ReLU(inplace=True)
            ),
            has_feedback=True,
            random_init=False
        )
        
        # PCoder 4: Layer 4 (512) predicts Layer 3 (256)
        self.pcoder4 = PCoder(
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=14, stride=2, padding=6),
                nn.ReLU(inplace=True)
            ),
            has_feedback=True,
            random_init=False
        )
        
        # PCoder 5: Layer 5 (512) predicts Layer 4 (512)
        self.pcoder5 = PCoder(
            nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=14, stride=2, padding=6),
                nn.ReLU(inplace=True)
            ),
            has_feedback=False,
            random_init=False
        )
    
    def register_extraction_hooks(self):
        """Register forward hooks to extract intermediate representations"""
        def make_hook(layer_idx):
            def hook(module, input, output):
                if self.training:
                    self.representations[layer_idx] = output
                return output
            return hook
        
        # Register hooks on backbone
        for i, pos in enumerate(self.layer_positions):
            self.backbone.features[pos].register_forward_hook(make_hook(i))
    
    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config."""
        cfg = super(PredifySimCLR, PredifySimCLR).add_and_assert_specific_cfg(cfg)
        
        # Default predictive coding parameters
        if "method_kwargs" not in cfg:
            cfg.method_kwargs = omegaconf.DictConfig({})
        
        cfg.method_kwargs.timesteps = omegaconf.DictConfig(cfg.method_kwargs).get("timesteps", 4)
        cfg.method_kwargs.pred_loss_weight = omegaconf.DictConfig(cfg.method_kwargs).get("pred_loss_weight", 1.0)
        cfg.method_kwargs.ffm = omegaconf.DictConfig(cfg.method_kwargs).get("ffm", [0.4, 0.3, 0.2, 0.1, 0.1])
        cfg.method_kwargs.fbm = omegaconf.DictConfig(cfg.method_kwargs).get("fbm", [0.05, 0.05, 0.05, 0.0, 0.0])
        cfg.method_kwargs.erm = omegaconf.DictConfig(cfg.method_kwargs).get("erm", [0.001, 0.001, 0.001, 0.001, 0.001])
        cfg.method_kwargs.use_local_updates = omegaconf.DictConfig(cfg.method_kwargs).get("use_local_updates", True)
        
        # Temporal parameters
        cfg.method_kwargs.use_temporal_pairs = omegaconf.DictConfig(cfg.method_kwargs).get("use_temporal_pairs", True)
        cfg.method_kwargs.temporal_temperature = omegaconf.DictConfig(cfg.method_kwargs).get("temporal_temperature", 0.07)
        
        # PCoder gradient parameters
        cfg.method_kwargs.pcoder_grad_scale = omegaconf.DictConfig(cfg.method_kwargs).get("pcoder_grad_scale", [0.1, 0.08, 0.06, 0.04, 0.02])
        cfg.method_kwargs.enable_pcoder_grads = omegaconf.DictConfig(cfg.method_kwargs).get("enable_pcoder_grads", False)
        
        # Standard SimCLR parameters
        cfg.method_kwargs.proj_hidden_dim = omegaconf.DictConfig(cfg.method_kwargs).get("proj_hidden_dim", 2048)
        cfg.method_kwargs.proj_output_dim = omegaconf.DictConfig(cfg.method_kwargs).get("proj_output_dim", 128)
        cfg.method_kwargs.temperature = omegaconf.DictConfig(cfg.method_kwargs).get("temperature", 0.1)
        
        return cfg
    
    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Add PCoder parameters to learnable params"""
        params = super().learnable_params
        
        # Add PCoder parameters
        for i in range(1, 6):
            params.append({
                "name": f"pcoder{i}",
                "params": getattr(self, f"pcoder{i}").parameters(),
            })
        
        # Add projector parameters
        params.append({
            "name": "projector",
            "params": self.projector.parameters(),
        })
        
        return params
    
    def reset_pcoders(self):
        """Reset PCoder states for new batch"""
        for i in range(1, 6):
            getattr(self, f"pcoder{i}").reset()
    
    def run_predictive_dynamics(self, target_reps: Dict[int, torch.Tensor], 
                              input_image: torch.Tensor = None) -> List[List[torch.Tensor]]:
        """Run predictive dynamics for specified timesteps
        
        Args:
            target_reps: Target representations for prediction (from another view or time)
            input_image: Target input image for PCoder1
        """
        all_errors = []
        
        # Use 1 timestep in reconstruction-only mode
        num_timesteps = 1 if not self.use_local_updates else self.timesteps
        
        for t in range(num_timesteps):
            timestep_errors = []
            
            # Handle PCoder1 - predicts input image
            if input_image is not None and 0 in self.representations:
                pcoder1 = self.pcoder1
                
                # ALWAYS use the full representation, detached
                ff_input = self.representations[0].detach()
                target = input_image.detach()
                
                fb_input = None
                if hasattr(self.pcoder2, 'prd') and self.pcoder2.prd is not None:
                    fb_input = self.pcoder2.prd
                
                try:
                    rep, pred = pcoder1(
                        ff=ff_input,
                        fb=fb_input,
                        target=target,
                        build_graph=self.enable_pcoder_grads and len(self.pcoder_grad_scale) > 0 and self.pcoder_grad_scale[0] > 0,  # Only build graph if grads enabled and scale > 0
                        ffm=self.ffm[0],
                        fbm=self.fbm[0] if fb_input is not None else 0.0,
                        erm=self.erm[0]
                    )
                    
                    # Scale the error AFTER computing it
                    if self.enable_pcoder_grads and len(self.pcoder_grad_scale) > 0 and self.pcoder_grad_scale[0] > 0:
                        scaled_error = pcoder1.prediction_error * self.pcoder_grad_scale[0]
                        timestep_errors.append(scaled_error)
                    else:
                        # No gradient contribution
                        timestep_errors.append(pcoder1.prediction_error.detach())
                    
                    if t < num_timesteps - 1 and self.use_local_updates:
                        self.representations[0] = rep
                        
                except Exception as e:
                    pass
            
            # Handle PCoders 2-5: predict from target representations
            for pcoder_num in range(2, 6):
                pcoder = getattr(self, f"pcoder{pcoder_num}")
                
                query_layer_idx = pcoder_num - 1
                target_layer_idx = pcoder_num - 2
                
                if query_layer_idx not in self.representations:
                    continue
                if target_layer_idx not in target_reps:
                    continue
                
                # ALWAYS use full representation, detached
                ff_input = self.representations[query_layer_idx].detach()
                target = target_reps[target_layer_idx].detach()
                
                fb_input = None
                if pcoder_num < 5:
                    higher_pcoder = getattr(self, f"pcoder{pcoder_num + 1}")
                    if hasattr(higher_pcoder, 'prd') and higher_pcoder.prd is not None:
                        fb_input = higher_pcoder.prd
                
                try:
                    rep, pred = pcoder(
                        ff=ff_input,
                        fb=fb_input,
                        target=target,
                        build_graph=self.enable_pcoder_grads and len(self.pcoder_grad_scale) > pcoder_num-1 and self.pcoder_grad_scale[pcoder_num-1] > 0,
                        ffm=self.ffm[pcoder_num-1],
                        fbm=self.fbm[pcoder_num-1] if fb_input is not None else 0.0,
                        erm=self.erm[pcoder_num-1]
                    )
                    
                    # Scale the error AFTER computing it
                    if self.enable_pcoder_grads and len(self.pcoder_grad_scale) > pcoder_num-1 and self.pcoder_grad_scale[pcoder_num-1] > 0:
                        scaled_error = pcoder.prediction_error * self.pcoder_grad_scale[pcoder_num-1]
                        timestep_errors.append(scaled_error)
                    else:
                        # No gradient contribution
                        timestep_errors.append(pcoder.prediction_error.detach())
                    
                    if t < num_timesteps - 1 and self.use_local_updates:
                        self.representations[query_layer_idx] = rep
                        
                except Exception as e:
                    continue
            
            all_errors.append(timestep_errors)
        
        return all_errors
    
    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Forward with predictive dynamics."""
        self.representations.clear()
        self.reset_pcoders()
        
        # Forward through backbone (hooks populate representations)
        spatial_features = self.backbone.features(X)
        
        # Get flattened features for projector
        pooled_features = self.backbone.avgpool(spatial_features)
        flattened_features = torch.flatten(pooled_features, 1)
        
        # Project features
        z = self.projector(flattened_features)
        
        # Determine features for KNN
        feats_for_knn = flattened_features
        if self.use_projected_features_for_knn and not self.training:
            feats_for_knn = z
        
        # Online classifier if available
        logits = None
        if hasattr(self, 'classifier'):
            logits = self.classifier(flattened_features.detach())
        
        return {
            "feats": feats_for_knn,
            "z": z,
            "logits": logits,
            "representations": self.representations.copy()
        }
    
    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step with predictive and contrastive losses"""
        indexes, X, targets = batch
        
        # Simplified temporal handling - dataset already provides pairs
        # X should be the result of transform applied to temporal pairs
        if not isinstance(X, list):
            X = [X]  # Convert single tensor to list format
        
        # Process all views
        outs = []
        all_representations = []
        
        for view in X[:self.num_large_crops]:
            out = self(view)
            outs.append(out)
            all_representations.append(out["representations"])
        
        # Compute losses
        total_loss = 0
        
        # 1. Predictive loss with detailed logging
        pred_loss = 0
        num_pred_errors = 0
        
        # Initialize PCoder-specific loss tracking
        pcoder_losses = {f"pcoder{i+1}": 0.0 for i in range(5)}
        pcoder_counts = {f"pcoder{i+1}": 0 for i in range(5)}
        
        # Track timesteps properly
        all_timesteps = []  # List of timesteps from prediction errors
        
        if len(outs) >= 2 and self.pred_loss_weight > 0:
            # Use second view's representations as targets for first view
            target_reps = all_representations[1]
            target_input = X[1] if len(X) > 1 else None
            
            # Run predictive dynamics
            self.representations = all_representations[0].copy()
            pred_errors = self.run_predictive_dynamics(target_reps, target_input)
            
            # Process prediction errors with detailed tracking
            for t, timestep_errors in enumerate(pred_errors):
                timestep_loss = 0
                for i, error in enumerate(timestep_errors):
                    pred_loss += error
                    num_pred_errors += 1
                    
                    # Track individual PCoder losses
                    pcoder_name = f"pcoder{i+1}"
                    pcoder_losses[pcoder_name] += error.item()
                    pcoder_counts[pcoder_name] += 1
                    timestep_loss += error.item()
                
                avg_timestep_loss = timestep_loss / len(timestep_errors) if timestep_errors else 0
                all_timesteps.append(avg_timestep_loss)
            
            if num_pred_errors > 0:
                pred_loss = pred_loss / num_pred_errors
                total_loss += self.pred_loss_weight * pred_loss
        
        # 2. Contrastive loss
        contrastive_loss = 0
        if len(outs) >= 2:
            # Get all projected features
            z = torch.cat([out["z"] for out in outs])
            
            # Compute SimCLR loss
            n_augs = len(outs)
            repeated_indexes = indexes.repeat(n_augs)
            
            contrastive_loss = simclr_loss_func(
                z,
                indexes=repeated_indexes,
                temperature=self.temperature,
            )
            total_loss += contrastive_loss
        
        # 3. Classification loss (if applicable)
        class_loss = 0
        if hasattr(self, 'classifier') and outs[0]["logits"] is not None:
            logits = torch.cat([out["logits"] for out in outs if out["logits"] is not None])
            targets_repeated = targets.repeat(len([out for out in outs if out["logits"] is not None]))
            class_loss = F.cross_entropy(logits, targets_repeated.long(), ignore_index=-1)
            total_loss += class_loss
        
        # Enhanced logging with detailed metrics
        local_updates_used = self.use_local_updates and len(outs) >= 2 and self.pred_loss_weight > 0
        
        metrics = {
            "train_total_loss": total_loss,
            "train_using_local_updates": float(local_updates_used),
            "train_reconstruction_only": float(not local_updates_used),
            "train_num_pred_errors": float(num_pred_errors),
            "train_pcoder_grads_enabled": float(self.enable_pcoder_grads),
        }
        
        # Log individual PCoder gradient scales
        if self.enable_pcoder_grads:
            for i in range(min(5, len(self.pcoder_grad_scale))):
                metrics[f"train_pcoder{i+1}_grad_scale"] = self.pcoder_grad_scale[i]
        
        # Log overall predictive loss
        if num_pred_errors > 0:
            metrics["train_pred_loss"] = pred_loss
            metrics["train_pred_loss_weighted"] = self.pred_loss_weight * pred_loss
            
            # Log individual PCoder losses
            for pcoder_name, loss_sum in pcoder_losses.items():
                if pcoder_counts[pcoder_name] > 0:
                    avg_loss = loss_sum / pcoder_counts[pcoder_name]
                    metrics[f"train_{pcoder_name}_loss"] = avg_loss
                else:
                    metrics[f"train_{pcoder_name}_loss"] = 0.0
            
            # Log timestep progression
            if all_timesteps:
                for t, ts_loss in enumerate(all_timesteps):
                    metrics[f"train_timestep_{t+1}_loss"] = ts_loss
                
                # Log timestep trends
                metrics["train_first_timestep_loss"] = all_timesteps[0]
                metrics["train_last_timestep_loss"] = all_timesteps[-1]
                if len(all_timesteps) > 1:
                    metrics["train_timestep_improvement"] = all_timesteps[0] - all_timesteps[-1]
        else:
            # Log zeros when no prediction errors are computed
            metrics["train_pred_loss"] = 0.0
            metrics["train_pred_loss_weighted"] = 0.0
            for i in range(1, 6):
                metrics[f"train_pcoder{i}_loss"] = 0.0
        
        # Log contrastive loss
        if len(outs) >= 2:
            metrics["train_contrastive_loss"] = contrastive_loss
        
        # Log classification loss
        if class_loss > 0:
            metrics["train_class_loss"] = class_loss
        
        # Log loss composition ratios
        if total_loss > 0:
            if num_pred_errors > 0:
                metrics["train_pred_loss_ratio"] = (self.pred_loss_weight * pred_loss) / total_loss
            if len(outs) >= 2:
                metrics["train_contrastive_loss_ratio"] = contrastive_loss / total_loss
            if class_loss > 0:
                metrics["train_class_loss_ratio"] = class_loss / total_loss
        
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        
        return total_loss