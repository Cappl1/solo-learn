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
        
        # True Predify parameters
        self.use_true_predify: bool = cfg.method_kwargs.get("use_true_predify", False)
        self.beta: List[float] = cfg.method_kwargs.get("beta", [0.4, 0.3, 0.2, 0.1, 0.1])
        self.lambda_: List[float] = cfg.method_kwargs.get("lambda_", [0.05, 0.05, 0.05, 0.05, 0.0])
        self.alpha: List[float] = cfg.method_kwargs.get("alpha", [0.001, 0.001, 0.001, 0.001, 0.001])
        self.true_predify_detach_errors: bool = cfg.method_kwargs.get("true_predify_detach_errors", True)
        self.true_predify_momentum: float = cfg.method_kwargs.get("true_predify_momentum", 0.9)
        
        # Ensure beta + lambda <= 1 for each layer
        for i in range(len(self.beta)):
            if i < len(self.lambda_):
                assert self.beta[i] + self.lambda_[i] <= 1.0, f"beta + lambda must be <= 1 for layer {i}: {self.beta[i]} + {self.lambda_[i]} = {self.beta[i] + self.lambda_[i]}"
        
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
        
        # Create partial forward functions for each layer segment
        if self.use_true_predify:
            self.create_layer_forward_functions()
        
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
        
        # Storage for evolved representations and trajectories
        self.evolved_representations = {}
        self.representation_trajectories = {}
        self.hooks_enabled = True
        
        # Debug mode for detailed logging
        self._debug_mode = False
        
        # Register hooks for feature extraction
        self.register_extraction_hooks()
    
    def create_layer_forward_functions(self):
        """Create functions to forward through specific layer segments"""
        self.layer_forward_funcs = []
        
        for i, layer_pos in enumerate(self.layer_positions):
            if i == 0:
                # From input to first hook position
                def make_forward(end_pos):
                    def forward_func(x):
                        return self.backbone.features[:end_pos+1](x)
                    return forward_func
                self.layer_forward_funcs.append(make_forward(layer_pos))
            else:
                # From previous hook to current hook
                def make_forward(start_pos, end_pos):
                    def forward_func(x):
                        return self.backbone.features[start_pos+1:end_pos+1](x)
                    return forward_func
                prev_pos = self.layer_positions[i-1]
                self.layer_forward_funcs.append(make_forward(prev_pos, layer_pos))
    
    def set_hooks_enabled(self, enabled: bool):
        """Enable or disable representation extraction hooks"""
        self.hooks_enabled = enabled
    
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
                if self.training and self.hooks_enabled:
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
        
        # True Predify parameters
        cfg.method_kwargs.use_true_predify = omegaconf.DictConfig(cfg.method_kwargs).get("use_true_predify", False)
        cfg.method_kwargs.beta = omegaconf.DictConfig(cfg.method_kwargs).get("beta", [0.4, 0.3, 0.2, 0.1, 0.1])
        cfg.method_kwargs.lambda_ = omegaconf.DictConfig(cfg.method_kwargs).get("lambda_", [0.05, 0.05, 0.05, 0.05, 0.0])
        cfg.method_kwargs.alpha = omegaconf.DictConfig(cfg.method_kwargs).get("alpha", [0.001, 0.001, 0.001, 0.001, 0.001])
        cfg.method_kwargs.true_predify_detach_errors = omegaconf.DictConfig(cfg.method_kwargs).get("true_predify_detach_errors", True)
        cfg.method_kwargs.true_predify_momentum = omegaconf.DictConfig(cfg.method_kwargs).get("true_predify_momentum", 0.9)
        
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
    
    def enable_debug_mode(self):
        """Enable debug mode for detailed logging of predictive dynamics"""
        self._debug_mode = True
    
    def disable_debug_mode(self):
        """Disable debug mode"""
        self._debug_mode = False
    
    def run_predictive_dynamics(self, target_reps: Dict[int, torch.Tensor], 
                              input_image: torch.Tensor = None, debug: bool = False) -> List[List[torch.Tensor]]:
        """Run predictive dynamics for specified timesteps
        
        Args:
            target_reps: Target representations for prediction (from another view or time)
            input_image: Target input image for PCoder1
            debug: Whether to print detailed debugging information
        """
        all_errors = []
        
        # Use 1 timestep in reconstruction-only mode
        num_timesteps = 1 if not self.use_local_updates else self.timesteps
        
        # Store original representations for the first timestep
        original_reps = self.representations.copy()
        
        for t in range(num_timesteps):
            timestep_errors = []
            
            if debug:
                print(f"\n--- Timestep {t} ---")
            
            # Handle PCoder1 - predicts input image
            if input_image is not None and 0 in original_reps:
                pcoder1 = self.pcoder1
                
                # Use original representation as feedforward input
                ff_input = original_reps[0].detach()
                target = input_image.detach()
                
                fb_input = None
                if hasattr(self.pcoder2, 'prd') and self.pcoder2.prd is not None:
                    fb_input = self.pcoder2.prd
                
                try:
                    rep, pred = pcoder1(
                        ff=ff_input,
                        fb=fb_input,
                        target=target,
                        build_graph=self.enable_pcoder_grads and len(self.pcoder_grad_scale) > 0 and self.pcoder_grad_scale[0] > 0,
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
                    
                    if debug:
                        print(f"  PCoder1 error: {pcoder1.prediction_error.item():.6f}")
                        
                except Exception as e:
                    if debug:
                        print(f"  PCoder1 failed: {e}")
                    pass
            
            # Handle PCoders 2-5: predict from target representations
            for pcoder_num in range(2, 6):
                pcoder = getattr(self, f"pcoder{pcoder_num}")
                
                query_layer_idx = pcoder_num - 1
                target_layer_idx = pcoder_num - 2
                
                if query_layer_idx not in original_reps:
                    continue
                if target_layer_idx not in target_reps:
                    continue
                
                # Use original representations as feedforward input
                ff_input = original_reps[query_layer_idx].detach()
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
                    
                    if debug:
                        print(f"  PCoder{pcoder_num} error: {pcoder.prediction_error.item():.6f}")
                        if hasattr(pcoder, 'rep') and pcoder.rep is not None:
                            print(f"  PCoder{pcoder_num} rep norm: {pcoder.rep.norm().item():.6f}")
                        
                except Exception as e:
                    if debug:
                        print(f"  PCoder{pcoder_num} failed: {e}")
                    continue
            
            # Log timestep error progression for debugging
            if debug and timestep_errors:
                avg_error = sum(err.item() if hasattr(err, 'item') else err for err in timestep_errors) / len(timestep_errors)
                print(f"  Average error at timestep {t}: {avg_error:.6f}")
            
            all_errors.append(timestep_errors)
        
        return all_errors
    
    def run_true_predictive_dynamics(self, input_image: torch.Tensor, 
                                   target_reps: Dict[int, torch.Tensor] = None,
                                   debug: bool = False) -> Dict[int, torch.Tensor]:
        """Run true predictive dynamics where backbone representations evolve
        
        Args:
            input_image: Input image
            target_reps: Optional target representations from another view
            debug: Whether to print debug information
            
        Returns:
            Dictionary of evolved representations at final timestep
        """
        # Initialize with standard forward pass representations
        with torch.no_grad():
            # Temporarily disable hooks to get clean forward pass
            self.set_hooks_enabled(False)
            _ = self.backbone.features(input_image)
            self.set_hooks_enabled(True)
        
        # e_n(t) - encoding layer representations that will evolve
        e = {i: self.representations[i].clone() for i in range(len(self.layer_positions))}
        
        # Enhanced tracking for detailed logging
        representation_trajectories = {i: [e[i].clone()] for i in e.keys()}
        
        # Track individual component contributions
        component_contributions = {
            'feedforward': {i: [] for i in range(len(self.layer_positions))},
            'feedback': {i: [] for i in range(len(self.layer_positions))},
            'memory': {i: [] for i in range(len(self.layer_positions))},
            'error_correction': {i: [] for i in range(len(self.layer_positions))}
        }
        
        # Track norms and changes
        norm_trajectories = {i: [e[i].norm().item()] for i in e.keys()}
        change_per_timestep = {i: [] for i in e.keys()}
        
        # Run dynamics for T timesteps
        for t in range(self.timesteps):
            if debug:
                print(f"\n--- True Predify Timestep {t} ---")
            
            # Will store e_n(t+1)
            e_next = {}
            
            # Update each layer
            for n in range(len(self.layer_positions)):
                pcoder = getattr(self, f"pcoder{n+1}")
                
                # 1. Feedforward term: β_n * W^f * e_{n-1}(t+1)
                if n == 0:
                    # First layer: feedforward from input (which is constant)
                    feedforward_input = input_image
                else:
                    # Use UPDATED representation from layer below
                    # Note: we use e_next[n-1] if available, else e[n-1]
                    feedforward_input = e_next.get(n-1, e[n-1])
                
                # Forward through this layer's weights
                with torch.no_grad():
                    feedforward_term = self.layer_forward_funcs[n](feedforward_input)
                feedforward_term = self.beta[n] * feedforward_term
                
                # 2. Feedback term: λ_n * d_n(t)
                # First, run PCoder to get predictions
                if n < len(self.layer_positions) - 1:
                    # Has feedback from above
                    fb = getattr(self, f"pcoder{n+2}").prd if hasattr(getattr(self, f"pcoder{n+2}"), 'prd') else None
                else:
                    fb = None
                
                # Determine target for PCoder
                if n == 0:
                    pcoder_target = input_image
                else:
                    pcoder_target = e[n-1] if target_reps is None else target_reps.get(n-1, e[n-1])
                
                # Run PCoder to get prediction
                _, _ = pcoder(
                    ff=e[n].detach() if self.true_predify_detach_errors else e[n],
                    fb=fb,
                    target=pcoder_target.detach() if self.true_predify_detach_errors else pcoder_target,
                    build_graph=not self.true_predify_detach_errors,
                    ffm=self.ffm[n],
                    fbm=self.fbm[n] if fb is not None else 0.0,
                    erm=self.erm[n]
                )
                
                # Get feedback correction (this is d_n(t) in the paper)
                if n < len(self.layer_positions) - 1 and fb is not None:
                    feedback_term = self.lambda_[n] * fb
                else:
                    feedback_term = torch.zeros_like(e[n])
                
                # 3. Memory term: (1 - β_n - λ_n) * e_n(t)
                memory_coefficient = 1 - self.beta[n] - self.lambda_[n]
                memory_term = memory_coefficient * e[n]
                
                # 4. Error correction term: -α_n * ∇ε_{n-1}(t)
                if n > 0 and hasattr(pcoder, 'prediction_error'):
                    # Compute gradient of prediction error w.r.t current representation
                    if self.true_predify_detach_errors:
                        # Simple error-based correction without gradients
                        error_correction = self.alpha[n] * pcoder.prediction_error.detach()
                    else:
                        # Full gradient-based correction (more expensive)
                        try:
                            error_grad = torch.autograd.grad(
                                pcoder.prediction_error.mean(),
                                e[n],
                                retain_graph=True,
                                create_graph=False
                            )[0]
                            error_correction = self.alpha[n] * error_grad
                        except RuntimeError:
                            # Fallback if gradient computation fails
                            error_correction = self.alpha[n] * pcoder.prediction_error.detach()
                else:
                    error_correction = torch.zeros_like(e[n])
                
                # Store component contributions for logging
                component_contributions['feedforward'][n].append(feedforward_term.norm().item())
                component_contributions['feedback'][n].append(feedback_term.norm().item() if isinstance(feedback_term, torch.Tensor) else 0.0)
                component_contributions['memory'][n].append(memory_term.norm().item())
                component_contributions['error_correction'][n].append(error_correction.norm().item() if isinstance(error_correction, torch.Tensor) else 0.0)
                
                # Combine all terms (Equation 2 from paper)
                e_next[n] = feedforward_term + feedback_term + memory_term - error_correction
                
                # Optional: Add momentum for stability
                if self.true_predify_momentum < 1.0:
                    e_next[n] = self.true_predify_momentum * e_next[n] + (1 - self.true_predify_momentum) * e[n]
                
                # Store trajectory and compute changes
                representation_trajectories[n].append(e_next[n].clone())
                norm_trajectories[n].append(e_next[n].norm().item())
                
                # Compute change from previous timestep
                change = (e_next[n] - e[n]).norm().item()
                relative_change = change / (e[n].norm().item() + 1e-6)
                change_per_timestep[n].append(relative_change)
                
                if debug:
                    print(f"  Layer {n}:")
                    print(f"    Feedforward contribution: {feedforward_term.norm().item():.4f}")
                    if isinstance(feedback_term, torch.Tensor):
                        print(f"    Feedback contribution: {feedback_term.norm().item():.4f}")
                    print(f"    Memory contribution: {memory_term.norm().item():.4f}")
                    if isinstance(error_correction, torch.Tensor):
                        print(f"    Error correction: {error_correction.norm().item():.4f}")
                    print(f"    Change this timestep: {relative_change:.6f}")
                    print(f"    Final norm: {e_next[n].norm().item():.4f}")
            
            # Update representations for next timestep
            e = e_next
        
        # Store trajectories and component analysis for logging
        self.representation_trajectories = representation_trajectories
        self.component_contributions = component_contributions
        self.norm_trajectories = norm_trajectories
        self.change_per_timestep = change_per_timestep
        
        return e
    
    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Forward with optional true predictive dynamics"""
        # Always use standard forward unless explicitly calling forward_with_true_predify
        return self.forward_standard(X)
    
    def forward_standard(self, X: torch.Tensor) -> Dict[str, Any]:
        """Standard forward pass (original implementation)"""
        self.representations.clear()
        # Only reset PCoders here for new batch, not in run_predictive_dynamics
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
    
    def forward_with_true_predify(self, X: torch.Tensor) -> Dict[str, Any]:
        """Forward pass using true predictive dynamics"""
        self.representations.clear()
        self.evolved_representations.clear()
        self.reset_pcoders()
        
        # Initial forward pass to populate representations
        with torch.no_grad():
            spatial_features = self.backbone.features(X)
        
        # Run true predictive dynamics
        evolved_reps = self.run_true_predictive_dynamics(
            input_image=X,
            target_reps=None,  # No target for single view
            debug=self._debug_mode
        )
        
        # Store evolved representations
        self.evolved_representations = evolved_reps
        
        # Continue forward from the last evolved representation
        last_layer_idx = len(self.layer_positions) - 1
        last_layer_pos = self.layer_positions[last_layer_idx]
        
        # Forward through remaining layers after last hook
        with torch.no_grad():
            x = evolved_reps[last_layer_idx]
            x = self.backbone.features[last_layer_pos+1:](x)
        
        # Pool and project
        pooled_features = self.backbone.avgpool(x)
        flattened_features = torch.flatten(pooled_features, 1)
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
            "representations": self.representations.copy(),
            "evolved_representations": evolved_reps
        }
    
    def _complete_forward_from_evolved_reps(self, input_image: torch.Tensor, 
                                          evolved_reps: Dict[int, torch.Tensor]) -> Dict[str, Any]:
        """Complete forward pass from evolved representations"""
        # Forward from last evolved representation
        last_layer_idx = len(self.layer_positions) - 1
        last_layer_pos = self.layer_positions[last_layer_idx]
        
        with torch.no_grad():
            x = evolved_reps[last_layer_idx]
            x = self.backbone.features[last_layer_pos+1:](x)
        
        pooled_features = self.backbone.avgpool(x)
        flattened_features = torch.flatten(pooled_features, 1)
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
            "representations": self.representations.copy(),
            "evolved_representations": evolved_reps
        }
    
    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step with true predictive dynamics and contrastive losses"""
        indexes, X, targets = batch
        
        # Simplified temporal handling - dataset already provides pairs
        # X should be the result of transform applied to temporal pairs
        if not isinstance(X, list):
            X = [X]  # Convert single tensor to list format
        
        # Process all views
        outs = []
        all_representations = []
        all_evolved_representations = []
        
        # Process views with optional true dynamics
        if self.use_true_predify and len(X) >= 2:
            # Simplified true dynamics: run dynamics between view1 and view2
            
            # Forward both views normally first
            view1, view2 = X[0], X[1]
            
            # Get initial representations for both views
            out1 = self.forward_standard(view1)
            out2 = self.forward_standard(view2)
            
            # Now run true dynamics: use view2's representations as targets for view1
            self.representations = out1["representations"].copy()
            evolved_reps = self.run_true_predictive_dynamics(
                input_image=view1,
                target_reps=out2["representations"],
                debug=self._debug_mode
            )
            
            # Create output with evolved representations
            out1_evolved = self._complete_forward_from_evolved_reps(view1, evolved_reps)
            
            outs = [out1_evolved, out2]
            all_representations = [out1_evolved["representations"], out2["representations"]]
            all_evolved_representations = [evolved_reps]
            
        else:
            # Standard forward for all views
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
        
        # Track representation changes for debugging local updates
        representation_changes = []
        
        if len(outs) >= 2 and self.pred_loss_weight > 0:
            # For true predify mode, losses are computed during dynamics
            if self.use_true_predify:
                # Extract prediction errors from PCoders after dynamics
                for i in range(1, 6):
                    pcoder = getattr(self, f"pcoder{i}")
                    if hasattr(pcoder, 'prediction_error') and pcoder.prediction_error is not None:
                        error = pcoder.prediction_error
                        pred_loss += error
                        num_pred_errors += 1
                        
                        # Track individual PCoder losses
                        pcoder_name = f"pcoder{i}"
                        pcoder_losses[pcoder_name] += error.item()
                        pcoder_counts[pcoder_name] += 1
            else:
                # Original predictive loss computation
                # Use second view's representations as targets for first view
                target_reps = all_representations[1]
                target_input = X[1] if len(X) > 1 else None
                
                # Store original representations for change tracking
                original_reps = all_representations[0].copy()
                
                # Run predictive dynamics
                self.representations = all_representations[0].copy()
                pred_errors = self.run_predictive_dynamics(target_reps, target_input, debug=self._debug_mode)
                
                # Track representation changes if local updates are enabled
                if self.use_local_updates and self.timesteps > 1:
                    for layer_idx in original_reps:
                        if layer_idx in self.representations:
                            original_norm = original_reps[layer_idx].norm().item()
                            final_norm = self.representations[layer_idx].norm().item()
                            change = abs(final_norm - original_norm) / (original_norm + 1e-8)
                            representation_changes.append(change)
                
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
        local_updates_used = self.use_local_updates and len(outs) >= 2 and self.pred_loss_weight > 0 and not self.use_true_predify
        true_predify_used = self.use_true_predify and len(outs) >= 2 and self.pred_loss_weight > 0
        
        metrics = {
            "train_total_loss": total_loss,
            "train_num_pred_errors": float(num_pred_errors),
        }
        
        # Add metrics for representation evolution (true predify)
        if self.use_true_predify and all_evolved_representations:
            # Comprehensive activation dynamics logging
            if hasattr(self, 'representation_trajectories') and hasattr(self, 'component_contributions'):
                
                # 1. Overall trajectory analysis
                total_displacement = 0
                max_displacement = 0
                min_displacement = float('inf')
                layer_displacements = []
                
                for layer_idx, trajectory in self.representation_trajectories.items():
                    if len(trajectory) > 1:
                        initial = trajectory[0]
                        final = trajectory[-1]
                        displacement = (final - initial).norm() / (initial.norm() + 1e-6)
                        displacement_val = displacement.item()
                        
                        total_displacement += displacement_val
                        max_displacement = max(max_displacement, displacement_val)
                        min_displacement = min(min_displacement, displacement_val)
                        layer_displacements.append(displacement_val)
                        
                        # Log per-layer displacement
                        metrics[f"train_layer_{layer_idx}_displacement"] = displacement_val
                
                if layer_displacements:
                    metrics["train_avg_representation_change"] = total_displacement / len(layer_displacements)
                    metrics["train_max_representation_change"] = max_displacement
                    metrics["train_min_representation_change"] = min_displacement
                    metrics["train_num_changed_layers"] = len(layer_displacements)
                    metrics["train_displacement_std"] = torch.std(torch.tensor(layer_displacements)).item()
                
                # 2. Per-timestep dynamics
                if hasattr(self, 'change_per_timestep'):
                    timestep_changes = []
                    for t in range(self.timesteps):
                        timestep_change = 0
                        layer_count = 0
                        for layer_idx in range(len(self.layer_positions)):
                            if layer_idx in self.change_per_timestep and t < len(self.change_per_timestep[layer_idx]):
                                timestep_change += self.change_per_timestep[layer_idx][t]
                                layer_count += 1
                        
                        if layer_count > 0:
                            avg_timestep_change = timestep_change / layer_count
                            timestep_changes.append(avg_timestep_change)
                            metrics[f"train_timestep_{t+1}_avg_change"] = avg_timestep_change
                    
                    # Log timestep trends
                    if len(timestep_changes) > 1:
                        metrics["train_first_timestep_change"] = timestep_changes[0]
                        metrics["train_last_timestep_change"] = timestep_changes[-1]
                        metrics["train_timestep_change_decay"] = timestep_changes[0] - timestep_changes[-1]
                        
                        # Compute convergence metrics
                        change_velocity = []
                        for t in range(1, len(timestep_changes)):
                            velocity = timestep_changes[t] - timestep_changes[t-1]
                            change_velocity.append(velocity)
                        
                        if change_velocity:
                            metrics["train_avg_change_velocity"] = sum(change_velocity) / len(change_velocity)
                            metrics["train_converging"] = float(sum(v < 0 for v in change_velocity) > len(change_velocity) / 2)
                
                # 3. Component contribution analysis
                if hasattr(self, 'component_contributions'):
                    for component_name, component_data in self.component_contributions.items():
                        component_totals = []
                        for layer_idx in range(len(self.layer_positions)):
                            if layer_idx in component_data and component_data[layer_idx]:
                                # Average contribution across timesteps for this layer
                                layer_avg = sum(component_data[layer_idx]) / len(component_data[layer_idx])
                                component_totals.append(layer_avg)
                                
                                # Log per-layer component contributions
                                metrics[f"train_layer_{layer_idx}_{component_name}_contribution"] = layer_avg
                        
                        # Log overall component statistics
                        if component_totals:
                            metrics[f"train_avg_{component_name}_contribution"] = sum(component_totals) / len(component_totals)
                            metrics[f"train_max_{component_name}_contribution"] = max(component_totals)
                            metrics[f"train_min_{component_name}_contribution"] = min(component_totals)
                
                # 4. Norm trajectory analysis
                if hasattr(self, 'norm_trajectories'):
                    norm_changes = []
                    for layer_idx, norm_traj in self.norm_trajectories.items():
                        if len(norm_traj) > 1:
                            initial_norm = norm_traj[0]
                            final_norm = norm_traj[-1]
                            norm_change = (final_norm - initial_norm) / (initial_norm + 1e-6)
                            norm_changes.append(norm_change)
                            
                            # Log per-layer norm changes
                            metrics[f"train_layer_{layer_idx}_norm_change"] = norm_change
                            metrics[f"train_layer_{layer_idx}_final_norm"] = final_norm
                            
                            # Compute norm stability (how much norm varies across timesteps)
                            if len(norm_traj) > 2:
                                norm_std = torch.std(torch.tensor(norm_traj)).item()
                                norm_stability = norm_std / (sum(norm_traj) / len(norm_traj) + 1e-6)
                                metrics[f"train_layer_{layer_idx}_norm_stability"] = norm_stability
                    
                    if norm_changes:
                        metrics["train_avg_norm_change"] = sum(norm_changes) / len(norm_changes)
                        metrics["train_norm_growing_layers"] = float(sum(1 for nc in norm_changes if nc > 0.01))
                        metrics["train_norm_shrinking_layers"] = float(sum(1 for nc in norm_changes if nc < -0.01))
                
                # 5. Dynamics quality metrics
                # Check if dynamics are well-behaved
                if layer_displacements and hasattr(self, 'change_per_timestep'):
                    # Measure if changes are decreasing (converging)
                    converging_layers = 0
                    for layer_idx in range(len(self.layer_positions)):
                        if layer_idx in self.change_per_timestep and len(self.change_per_timestep[layer_idx]) > 1:
                            changes = self.change_per_timestep[layer_idx]
                            if len(changes) >= 2 and changes[-1] < changes[0]:
                                converging_layers += 1
                    
                    metrics["train_converging_layer_fraction"] = converging_layers / len(self.layer_positions)
                    metrics["train_dynamics_well_behaved"] = float(converging_layers >= len(self.layer_positions) / 2)
                    
                    # Overall dynamics "energy"
                    total_activity = sum(layer_displacements)
                    metrics["train_total_dynamics_activity"] = total_activity
                    
                    # Balance between components
                    if hasattr(self, 'component_contributions'):
                        ff_total = sum(self.component_contributions['feedforward'][i][-1] if self.component_contributions['feedforward'][i] else 0 
                                     for i in range(len(self.layer_positions)))
                        fb_total = sum(self.component_contributions['feedback'][i][-1] if self.component_contributions['feedback'][i] else 0 
                                     for i in range(len(self.layer_positions)))
                        mem_total = sum(self.component_contributions['memory'][i][-1] if self.component_contributions['memory'][i] else 0 
                                      for i in range(len(self.layer_positions)))
                        err_total = sum(self.component_contributions['error_correction'][i][-1] if self.component_contributions['error_correction'][i] else 0 
                                      for i in range(len(self.layer_positions)))
                        
                        total_components = ff_total + fb_total + mem_total + err_total
                        if total_components > 0:
                            metrics["train_feedforward_dominance"] = ff_total / total_components
                            metrics["train_feedback_dominance"] = fb_total / total_components
                            metrics["train_memory_dominance"] = mem_total / total_components
                            metrics["train_error_dominance"] = err_total / total_components
            
            else:
                # Fallback to simple metrics if detailed tracking not available
                metrics["train_avg_representation_change"] = 0.0
                metrics["train_max_representation_change"] = 0.0
                metrics["train_min_representation_change"] = 0.0
                metrics["train_num_changed_layers"] = 0
        else:
            # Log representation changes for debugging local updates (original code)
            if representation_changes:
                metrics["train_avg_rep_change"] = sum(representation_changes) / len(representation_changes)
                metrics["train_max_rep_change"] = max(representation_changes)
                metrics["train_min_rep_change"] = min(representation_changes)
                metrics["train_num_changed_layers"] = len(representation_changes)
            else:
                metrics["train_avg_rep_change"] = 0.0
                metrics["train_max_rep_change"] = 0.0
                metrics["train_min_rep_change"] = 0.0
                metrics["train_num_changed_layers"] = 0
        
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
        
        # Log overall predictive loss and PCoder-specific losses
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
            
            # Log timestep progression (for non-true predify mode)
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