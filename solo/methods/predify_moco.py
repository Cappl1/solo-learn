import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Sequence, Tuple
import omegaconf

from solo.methods.base import BaseMomentumMethod
from solo.utils.pcoder import PCoder
from solo.losses.mocov3 import mocov3_loss_func
from solo.utils.momentum import initialize_momentum_params


class PredifyMoCo(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements PredifyMoCo: Predictive Coding + Momentum Contrastive Learning.
        
        This method combines hierarchical predictive coding with momentum contrastive learning.
        Each layer of the query encoder predicts previous layers from the momentum encoder,
        running multiple timesteps of predictive dynamics before computing contrastive loss.

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of projected features.
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
                temperature (float): temperature for the softmax in the contrastive loss.
                timesteps (int): number of predictive dynamics timesteps.
                pred_loss_weight (float): weight for predictive loss.
                ffm (list): feedforward multipliers for each PCoder.
                fbm (list): feedback multipliers for each PCoder.
                erm (list): error multipliers for each PCoder.
                use_local_updates (bool): whether to use local updates.
                pcoder_grad_scale (list): scale for PCoder gradients.
                enable_pcoder_grads (bool): whether to enable PCoder gradients.
        """
        super().__init__(cfg)
        
        # Flag to control KNN feature type
        self.use_projected_features_for_knn = \
            "vgg" in self.backbone_name or \
            "efficientnet" in self.backbone_name # Add other backbones if needed
        
        # Predictive coding parameters
        self.timesteps: int = cfg.method_kwargs.timesteps
        self.pred_loss_weight: float = cfg.method_kwargs.pred_loss_weight
        self.use_local_updates: bool = cfg.method_kwargs.use_local_updates
        
        # PCoder gradient parameters
        self.pcoder_grad_scale: List[float] = cfg.method_kwargs.get("pcoder_grad_scale", [0.1, 0.08, 0.06, 0.04, 0.02])
        self.enable_pcoder_grads: bool = cfg.method_kwargs.get("enable_pcoder_grads", False)
        
        # PCoder multipliers
        self.ffm = cfg.method_kwargs.ffm  # [0.4, 0.3, 0.2, 0.1, 0.1]
        self.fbm = cfg.method_kwargs.fbm  # [0.05, 0.05, 0.05, 0.0, 0.0]
        self.erm = cfg.method_kwargs.erm  # [0.001, 0.001, 0.001, 0.001, 0.001]
        
        # Standard MoCo parameters
        self.temperature: float = cfg.method_kwargs.temperature
        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim
        
        # Only support VGG16 for now
        assert "vgg" in self.backbone_name, "PredifyMoCo currently only supports VGG16 backbone"
        
        # VGG16 layer positions and dimensions for PCoder hooks
        self.layer_positions = [3, 8, 15, 22, 29]  # ReLU after conv layers
        self.layer_dims = [64, 128, 256, 512, 512]
        
        # Create predictive coding modules
        self.create_pcoders()
        
        # Standard MoCo projector and predictor
        self.projector = self._build_mlp(
            2,
            self.features_dim,
            proj_hidden_dim,
            proj_output_dim,
        )
        self.momentum_projector = self._build_mlp(
            2,
            self.features_dim,
            proj_hidden_dim,
            proj_output_dim,
        )
        self.predictor = self._build_mlp(
            2,
            proj_output_dim,
            pred_hidden_dim,
            proj_output_dim,
            last_bn=False,
        )
        
        # Initialize momentum parameters
        initialize_momentum_params(self.projector, self.momentum_projector)
        
        # Store intermediate representations
        self.query_reps = {}
        self.momentum_reps = {}
        
        # Register hooks for feature extraction
        self.register_extraction_hooks()
    
    def create_pcoders(self):
        """Create PCoder modules for hierarchical cross-prediction"""
        
        # PCoder 1: Layer 1 (64) predicts INPUT IMAGE (3) from momentum
        # Uses larger kernel and stride=1 to match original PrediFy
        self.pcoder1 = PCoder(
            nn.Sequential(
                nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2),
                nn.Tanh()  # Use Tanh for image reconstruction
            ),
            has_feedback=True,
            random_init=False
        )
        
        # PCoder 2: Layer 2 (128) predicts Layer 1 (64) from momentum  
        # Uses larger kernel size like original PrediFy
        self.pcoder2 = PCoder(
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=10, stride=2, padding=4),
                nn.ReLU(inplace=True)
            ),
            has_feedback=True,
            random_init=False
        )
        
        # PCoder 3: Layer 3 (256) predicts Layer 2 (128) from momentum
        self.pcoder3 = PCoder(
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=10, stride=2, padding=4),
                nn.ReLU(inplace=True)
            ),
            has_feedback=True,
            random_init=False
        )
        
        # PCoder 4: Layer 4 (512) predicts Layer 3 (256) from momentum
        self.pcoder4 = PCoder(
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=14, stride=2, padding=6),
                nn.ReLU(inplace=True)
            ),
            has_feedback=True,
            random_init=False
        )
        
        # PCoder 5: Layer 5 (512) predicts Layer 4 (512) from momentum
        self.pcoder5 = PCoder(
            nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=14, stride=2, padding=6),
                nn.ReLU(inplace=True)
            ),
            has_feedback=False,  # Top layer has no feedback
            random_init=False
        )
    
    def register_extraction_hooks(self):
        """Register forward hooks to extract intermediate representations"""
        def make_hook(layer_idx):
            def hook(module, input, output):
                if self.training:
                    # Store query representations during training
                    self.query_reps[layer_idx] = output
                return output
            return hook
        
        def make_momentum_hook(layer_idx):
            def hook(module, input, output):
                if self.training:
                    # Store momentum representations during training
                    self.momentum_reps[layer_idx] = output.detach()
                return output
            return hook
        
        # Register hooks on query backbone
        for i, pos in enumerate(self.layer_positions):
            self.backbone.features[pos].register_forward_hook(make_hook(i))
            
        # Register hooks on momentum backbone
        for i, pos in enumerate(self.layer_positions):
            self.momentum_backbone.features[pos].register_forward_hook(make_momentum_hook(i))
    
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        """Build MLP projection head"""
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
    
    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config."""
        cfg = super(PredifyMoCo, PredifyMoCo).add_and_assert_specific_cfg(cfg)
        
        # Default predictive coding parameters
        if "method_kwargs" not in cfg:
            cfg.method_kwargs = omegaconf.DictConfig({})
        
        cfg.method_kwargs.timesteps = omegaconf.DictConfig(cfg.method_kwargs).get("timesteps", 4)
        cfg.method_kwargs.pred_loss_weight = omegaconf.DictConfig(cfg.method_kwargs).get("pred_loss_weight", 1.0)
        cfg.method_kwargs.ffm = omegaconf.DictConfig(cfg.method_kwargs).get("ffm", [0.4, 0.3, 0.2, 0.1, 0.1])
        cfg.method_kwargs.fbm = omegaconf.DictConfig(cfg.method_kwargs).get("fbm", [0.05, 0.05, 0.05, 0.0, 0.0])
        cfg.method_kwargs.erm = omegaconf.DictConfig(cfg.method_kwargs).get("erm", [0.001, 0.001, 0.001, 0.001, 0.001])
        cfg.method_kwargs.use_local_updates = omegaconf.DictConfig(cfg.method_kwargs).get("use_local_updates", True)
        
        # PCoder gradient parameters
        cfg.method_kwargs.pcoder_grad_scale = omegaconf.DictConfig(cfg.method_kwargs).get("pcoder_grad_scale", [0.1, 0.08, 0.06, 0.04, 0.02])
        cfg.method_kwargs.enable_pcoder_grads = omegaconf.DictConfig(cfg.method_kwargs).get("enable_pcoder_grads", False)
        
        # Standard MoCo parameters
        cfg.method_kwargs.proj_hidden_dim = omegaconf.DictConfig(cfg.method_kwargs).get("proj_hidden_dim", 4096)
        cfg.method_kwargs.proj_output_dim = omegaconf.DictConfig(cfg.method_kwargs).get("proj_output_dim", 256)
        cfg.method_kwargs.pred_hidden_dim = omegaconf.DictConfig(cfg.method_kwargs).get("pred_hidden_dim", 4096)
        cfg.method_kwargs.temperature = omegaconf.DictConfig(cfg.method_kwargs).get("temperature", 0.2)
        
        return cfg
    
    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Add PCoder parameters to learnable params"""
        params = super().learnable_params
        
        # Add PCoder parameters - always needed for reconstruction losses
        for i in range(1, 6):
            params.append({
                "name": f"pcoder{i}",
                "params": getattr(self, f"pcoder{i}").parameters(),
            })
        
        # Add query projector and predictor parameters - always needed
        params.append({
            "name": "projector",
            "params": self.projector.parameters(),
        })
        params.append({
            "name": "predictor", 
            "params": self.predictor.parameters(),
        })
        
        # Note: momentum_projector is NOT added to learnable params since it's 
        # updated via EMA only and never receives gradients directly
        
        return params
    
    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Define momentum pairs for EMA updates"""
        pairs = super().momentum_pairs
        pairs.append((self.projector, self.momentum_projector))
        return pairs
    
    def reset_pcoders(self):
        """Reset PCoder states for new batch"""
        for i in range(1, 6):
            getattr(self, f"pcoder{i}").reset()
    
    def run_predictive_dynamics(self, num_timesteps: int) -> List[List[torch.Tensor]]:
        """Run predictive dynamics for specified timesteps
        
        Note: All feedforward inputs are detached to prevent reconstruction gradients 
        from interfering with backbone learning. PCoders learn independently while
        backbone focuses on semantic representation learning via contrastive loss.
        """
        
        # FORCE 1 timestep in reconstruction-only mode, regardless of input
        if not self.use_local_updates:
            num_timesteps = 1
        
        all_errors = []
        
        for t in range(num_timesteps):
            timestep_errors = []
            
            # Handle PCoder1 specially - predicts input image from Layer 1
            if hasattr(self, 'query_input') and hasattr(self, 'momentum_input') and 0 in self.query_reps:
                pcoder1 = self.pcoder1
                
                # Layer 1 from query encoder predicts input image from momentum
                # Apply gradient scaling if enabled, otherwise detach
                if self.enable_pcoder_grads:
                    scale = self.pcoder_grad_scale[0] if len(self.pcoder_grad_scale) > 0 else 0.1
                    ff_input = self.query_reps[0] * scale  # Layer 1 (64 channels)
                else:
                    ff_input = self.query_reps[0].detach()  # Layer 1 (64 channels)
                target = self.momentum_input.detach()  # Input image (3 channels)
                
                # Feedback from PCoder2 (if available)
                fb_input = None
                if hasattr(self.pcoder2, 'prd') and self.pcoder2.prd is not None:
                    fb_input = self.pcoder2.prd
                
                try:
                    # PCoder1 forward
                    rep, pred = pcoder1(
                        ff=ff_input,
                        fb=fb_input,
                        target=target,
                        build_graph=True,
                        ffm=self.ffm[0],
                        fbm=self.fbm[0] if fb_input is not None else 0.0,
                        erm=self.erm[0]
                    )
                    
                    timestep_errors.append(pcoder1.prediction_error)
                    
                    # Update representation for next timestep ONLY if using local updates
                    if t < num_timesteps - 1 and self.use_local_updates:
                        self.query_reps[0] = rep
                        
                except Exception as e:
                    pass
            
            # Handle PCoders 2-5: predict previous layer from momentum encoder
            for pcoder_num in range(2, 6):  # PCoder numbers 2, 3, 4, 5
                pcoder = getattr(self, f"pcoder{pcoder_num}")
                
                # Calculate correct layer indices
                query_layer_idx = pcoder_num - 1  # PCoder2 uses layer 1, PCoder3 uses layer 2, etc.
                target_layer_idx = pcoder_num - 2  # PCoder2 targets layer 0, PCoder3 targets layer 1, etc.
                
                # Check if we have the required representations
                if query_layer_idx not in self.query_reps:
                    continue
                if target_layer_idx not in self.momentum_reps:
                    continue
                
                # Current layer from query encoder
                # Apply gradient scaling if enabled, otherwise detach
                if self.enable_pcoder_grads:
                    scale = self.pcoder_grad_scale[pcoder_num-1] if len(self.pcoder_grad_scale) > pcoder_num-1 else 0.1
                    ff_input = self.query_reps[query_layer_idx] * scale
                else:
                    ff_input = self.query_reps[query_layer_idx].detach()
                
                # Target: previous layer from momentum encoder (detached)
                target = self.momentum_reps[target_layer_idx].detach()
                
                # Feedback from higher layer (if exists)
                fb_input = None
                if pcoder_num < 5:  # PCoders 2, 3, 4 can have feedback from 3, 4, 5
                    higher_pcoder = getattr(self, f"pcoder{pcoder_num + 1}")
                    if hasattr(higher_pcoder, 'prd') and higher_pcoder.prd is not None:
                        fb_input = higher_pcoder.prd
                
                try:
                    # PCoder forward
                    rep, pred = pcoder(
                        ff=ff_input,
                        fb=fb_input,
                        target=target,
                        build_graph=True,
                        ffm=self.ffm[pcoder_num-1],
                        fbm=self.fbm[pcoder_num-1] if fb_input is not None else 0.0,
                        erm=self.erm[pcoder_num-1]
                    )
                    
                    timestep_errors.append(pcoder.prediction_error)
                    
                    # Update representation for next timestep ONLY if using local updates
                    if t < num_timesteps - 1 and self.use_local_updates:
                        self.query_reps[query_layer_idx] = rep
                        
                except Exception as e:
                    continue
            
            all_errors.append(timestep_errors)
        
        return all_errors
    
    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Forward with predictive dynamics and contrastive learning."""
        self.query_reps.clear() # query_reps are populated by hooks on self.backbone.features
        self.reset_pcoders()
        
        # Store the input image for PCoder1 target
        self.query_input = X
        
        # Step 1: Get initial spatial features by passing through the VGG features module.
        # Hooks on self.backbone.features will populate self.query_reps.
        spatial_features = self.backbone.features(X)

        current_feats_for_downstream = spatial_features # Default to initial spatial features

        if self.use_local_updates and hasattr(self, '_momentum_targets_available') and self._momentum_targets_available:
            # Run predictive dynamics. This will update self.query_reps internally.
            # The 'rep' attribute of each pcoder and self.query_reps themselves are updated.
            pred_errors = self.run_predictive_dynamics(self.timesteps)
            self._current_pred_errors = pred_errors # Store for loss computation
            
            # The top-most updated spatial representation is now in self.query_reps[4]
            # (or pcoder5.rep, which should be the same if dynamics ran)
            current_feats_for_downstream = self.query_reps[4] 
            # This ^ is the B x 512 x H x W updated feature map from the top of VGG features
        else:
            # Reconstruction-only mode: No local updates, features stay unchanged
            pass

        # Step 2: Apply VGG's original pooling and flatten to the (potentially updated) spatial features.
        # This ensures the features_dim (e.g., 25088) is consistent for projector and classifier.
        pooled_features = self.backbone.avgpool(current_feats_for_downstream)
        flattened_features = torch.flatten(pooled_features, 1) # Should be (B, 25088) for VGG
        
        # Now, flattened_features is the input for the projector & online classifier
        projected_feats_z = self.projector(flattened_features) # This is 'z'
        predicted_p = self.predictor(projected_feats_z)     # This is 'p'

        # Determine what to return as "feats" for KNN 
        feats_for_knn = flattened_features # Default to the 25088-dim features
        if self.use_projected_features_for_knn and not self.training:
            feats_for_knn = projected_feats_z # Use 128-dim for KNN during validation
        
        final_logits = None
        if hasattr(self, 'classifier'):
            # The online classifier (from BaseMethod) expects features_dim (25088 for VGG)
            final_logits = self.classifier(flattened_features.detach())

        return {
            "feats": feats_for_knn, 
            "z": projected_feats_z,
            "p": predicted_p,
            "logits": final_logits
        }
    
    def apply_local_updates(self, original_feats: torch.Tensor) -> torch.Tensor:
        """Apply PrediFy local update rule to replace query encoder activations"""
        if len(self.query_reps) != 5 or len(self.momentum_reps) != 5:
            return original_feats
        
        # Run predictive dynamics to get updated representations
        pred_errors = self.run_predictive_dynamics(self.timesteps)
        
        # Store the prediction errors for loss computation
        self._current_pred_errors = pred_errors
        
        # Get the final updated representation from the top PCoder
        # The representations have been updated through timesteps in run_predictive_dynamics
        updated_top_rep = getattr(self, 'pcoder5').rep if hasattr(getattr(self, 'pcoder5'), 'rep') and getattr(self, 'pcoder5').rep is not None else self.query_reps[4]
        
        # Apply global pooling to match the final feature dimensions
        if updated_top_rep.dim() == 4:  # Spatial features (B, C, H, W)
            # Apply adaptive average pooling to match original features shape
            updated_feats = F.adaptive_avg_pool2d(updated_top_rep, (1, 1)).flatten(1)
        else:
            updated_feats = updated_top_rep
        
        return updated_feats
    
    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Momentum forward for creating targets"""
        # Clear previous momentum representations
        self.momentum_reps.clear()
        
        # Store the input image for PCoder1 target
        self.momentum_input = X
        
        # Forward through momentum backbone (hooks will populate momentum_reps)
        feats = self.momentum_backbone(X)
        z = self.momentum_projector(feats)
        
        return {"feats": feats, "z": z}
    
    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step with predictive and contrastive losses"""
        _, X, targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X
        
        # Process both views
        outs = []
        momentum_outs = []
        
        for view_idx, view in enumerate(X[:self.num_large_crops]):
            # Clear representations for this view
            self.query_reps.clear()
            self.momentum_reps.clear()
            self._momentum_targets_available = False
            
            # STEP 1: First forward pass to get momentum targets
            with torch.no_grad():
                momentum_out = self.momentum_forward(view)
            
            # STEP 2: Check if we have momentum representations for local updates
            if len(self.momentum_reps) == 5:
                self._momentum_targets_available = True
            
            # STEP 3: Forward pass with potential local updates
            out = self(view)
            
            # STEP 4: Extract prediction errors from local updates (if applied)
            if hasattr(self, '_current_pred_errors'):
                out["pred_errors"] = self._current_pred_errors
                delattr(self, '_current_pred_errors')  # Clean up
            else:
                # Always run predictive dynamics to ensure PCoder params are used (prevents DDP errors)
                if len(self.query_reps) == 5 and len(self.momentum_reps) == 5:
                    # For reconstruction-only mode, use just 1 timestep
                    # For full PrediFy mode, use configured timesteps
                    timesteps_to_use = 1 if not self.use_local_updates else self.timesteps
                    pred_errors = self.run_predictive_dynamics(timesteps_to_use)
                    out["pred_errors"] = pred_errors
                else:
                    # Still run PCoders with available data to ensure params are used
                    if len(self.query_reps) > 0 and len(self.momentum_reps) > 0:
                        timesteps_to_use = 1 if not self.use_local_updates else self.timesteps
                        pred_errors = self.run_predictive_dynamics(timesteps_to_use)
                        out["pred_errors"] = pred_errors
                    else:
                        out["pred_errors"] = []
            
            outs.append(out)
            momentum_outs.append(momentum_out)
        
        # Compute losses
        total_loss = 0
        
        # 1. Predictive loss with detailed logging
        pred_loss = 0
        num_pred_errors = 0
        
        # Initialize PCoder-specific loss tracking
        pcoder_losses = {f"pcoder{i+1}": 0.0 for i in range(5)}
        pcoder_counts = {f"pcoder{i+1}": 0 for i in range(5)}
        
        # Track timesteps properly: collect timesteps from each view separately
        all_view_timesteps = []  # List of lists: each inner list is timesteps from one view
        
        for view_idx, out in enumerate(outs):
            if out["pred_errors"]:
                view_timesteps = []  # Timesteps for this specific view
                for t, timestep_errors in enumerate(out["pred_errors"]):
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
                    view_timesteps.append(avg_timestep_loss)
                
                all_view_timesteps.append(view_timesteps)
        
        # Compute averaged timestep losses across views
        timestep_losses = []
        if all_view_timesteps:
            max_timesteps = max(len(view_ts) for view_ts in all_view_timesteps)
            for t in range(max_timesteps):
                # Average timestep t across all views that have it
                timestep_t_losses = [view_ts[t] for view_ts in all_view_timesteps if t < len(view_ts)]
                avg_timestep_t_loss = sum(timestep_t_losses) / len(timestep_t_losses) if timestep_t_losses else 0
                timestep_losses.append(avg_timestep_t_loss)
        
        if num_pred_errors > 0:
            pred_loss = pred_loss / num_pred_errors
            total_loss += self.pred_loss_weight * pred_loss

        # 2. Contrastive loss (MoCo-style)
        contrastive_loss = 0
        if len(outs) >= 2:
            q1, q2 = outs[0]["p"], outs[1]["p"]
            k1, k2 = momentum_outs[0]["z"], momentum_outs[1]["z"]
            
            # Compute contrastive loss
            contrastive_loss = mocov3_loss_func(q1, k2, self.temperature) + \
                              mocov3_loss_func(q2, k1, self.temperature)
            contrastive_loss = contrastive_loss / 2
            total_loss += contrastive_loss

        # 3. Classification loss (if applicable)
        class_loss = 0
        if hasattr(self, 'classifier') and outs[0]["logits"] is not None:
            logits = torch.cat([out["logits"] for out in outs])
            targets_repeated = targets.repeat(len(outs))
            class_loss = F.cross_entropy(logits, targets_repeated.long(), ignore_index=-1)
            total_loss += class_loss

        # Enhanced logging with detailed metrics
        local_updates_used = self.use_local_updates and hasattr(self, '_momentum_targets_available') and self._momentum_targets_available
        
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
            if timestep_losses:
                for t, ts_loss in enumerate(timestep_losses):
                    metrics[f"train_timestep_{t+1}_loss"] = ts_loss
                
                # Log timestep trends
                metrics["train_first_timestep_loss"] = timestep_losses[0]
                metrics["train_last_timestep_loss"] = timestep_losses[-1]
                if len(timestep_losses) > 1:
                    metrics["train_timestep_improvement"] = timestep_losses[0] - timestep_losses[-1]
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
        
        self.log_dict(metrics, on_epoch=True, on_step=True, sync_dist=True)

        return total_loss 