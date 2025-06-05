# Copyright 2023 solo-learn development team.
# Additional contributions by the I-JEPA authors.
# This source code is licensed under the Apache Lincense 2.0 found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Sequence, Tuple

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
# We will need to adapt or import mask generation and application logic
# from src.masks.multiblock import MaskCollator as MBMaskCollator (from Meta I-JEPA)
# from src.masks.utils import apply_masks (from Meta I-JEPA)

import math

# Placeholder for mask generation utilities - we'll need to implement/adapt these
class IJEPAMaskGenerator:
    def __init__(self, input_size, patch_size, pred_mask_scale, enc_mask_scale, aspect_ratio, nenc, npred, allow_overlap, min_keep):
        # Store all parameters from MBMaskCollator
        self.input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
        self.patch_size = patch_size
        self.pred_mask_scale = pred_mask_scale
        self.enc_mask_scale = enc_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.allow_overlap = allow_overlap
        self.min_keep = min_keep
        
        self.height, self.width = self.input_size[0] // patch_size, self.input_size[1] // patch_size
        self._itr_counter = 0 # Simple counter for seeding, similar to original Value('i', -1)

    def _sample_block_size(self, generator: torch.Generator, scale: Tuple[float, float], aspect_ratio_scale: Tuple[float, float]):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio_val = min_ar + _rand * (max_ar - min_ar) # Renamed from aspect_ratio to avoid conflict
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio_val)))
        w = int(round(math.sqrt(max_keep / aspect_ratio_val)))
        # Ensure block dimensions are within image patch dimensions
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1
        # Ensure h and w are at least 1 if max_keep is small but positive
        if max_keep > 0:
            h = max(h, 1)
            w = max(w, 1)
        else:
            h = 0 # if max_keep is 0, block size is 0
            w = 0
        return (h, w)

    def _sample_block_mask(self, b_size: Tuple[int, int], device: torch.device, acceptable_regions: List[torch.Tensor] = None):
        h, w = b_size
        if h == 0 or w == 0: # If block size is zero, return empty mask and full complement
            return torch.empty((0), dtype=torch.long, device=device), torch.ones((self.height, self.width), dtype=torch.bool, device=device)

        def constrain_mask(mask_to_constrain, current_tries=0):
            N = max(int(len(acceptable_regions)-current_tries), 0)
            for k_region in range(N):
                mask_to_constrain *= acceptable_regions[k_region]
        
        tries = 0
        timeout = og_timeout = 20 # As in original I-JEPA
        valid_mask_found = False
        mask_indices = torch.empty((0), dtype=torch.long, device=device)
        
        # Variables to store top/left of the last valid attempt if loop finishes without valid mask
        last_top, last_left = 0, 0 

        while not valid_mask_found:
            last_top = torch.randint(0, self.height - h + 1, (1,), device=device).item() 
            last_left = torch.randint(0, self.width - w + 1, (1,), device=device).item() 
            
            current_mask_map = torch.zeros((self.height, self.width), dtype=torch.bool, device=device)
            current_mask_map[last_top:last_top+h, last_left:last_left+w] = True
            
            if acceptable_regions is not None:
                constrain_mask(current_mask_map, tries)
            
            mask_indices = torch.nonzero(current_mask_map.flatten()).squeeze(-1)
            
            valid_mask_found = len(mask_indices) >= self.min_keep
            
            if not valid_mask_found:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    if acceptable_regions and tries > len(acceptable_regions):
                        break 
            if acceptable_regions and tries > len(acceptable_regions) and not valid_mask_found: 
                 break
            elif not acceptable_regions and timeout == 0 and not valid_mask_found: # if no acceptable_regions, just try a few times
                if tries > og_timeout : # Allow limited retries even without acceptable_regions
                    break

        mask_complement_map = torch.ones((self.height, self.width), dtype=torch.bool, device=device)
        # Use last_top, last_left for the complement map, corresponding to the last attempt
        # irrespective of whether it was 'valid_mask_found' for min_keep, to define the block area.
        mask_complement_map[last_top:last_top+h, last_left:last_left+w] = False
        
        return mask_indices, mask_complement_map

    def __call__(self, device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        self._itr_counter += 1
        seed = self._itr_counter 
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        pred_block_shape = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio
        )
        enc_block_shape = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.) 
        )

        masks_pred_for_image = []
        list_of_pred_mask_complements = []
        for _ in range(self.npred):
            mask_p_indices, mask_p_complement_map = self._sample_block_mask(pred_block_shape, device=device)
            masks_pred_for_image.append(mask_p_indices)
            list_of_pred_mask_complements.append(mask_p_complement_map)

        masks_enc_for_image = []
        current_acceptable_regions_for_enc = None
        if not self.allow_overlap:
            if list_of_pred_mask_complements:
                intersected_complement = list_of_pred_mask_complements[0]
                for i in range(1, len(list_of_pred_mask_complements)):
                    intersected_complement = intersected_complement & list_of_pred_mask_complements[i]
                current_acceptable_regions_for_enc = [intersected_complement] 
        
        for _ in range(self.nenc):
            mask_e_indices, _ = self._sample_block_mask(enc_block_shape, device=device, acceptable_regions=current_acceptable_regions_for_enc)
            masks_enc_for_image.append(mask_e_indices)
            
        return masks_enc_for_image, masks_pred_for_image

def apply_ijepa_masks(
    features: torch.Tensor, # Shape: [B, NumTotalPatches, DimFeatures]
    batched_mask_indices: torch.Tensor # Shape: [B, NumMasksToApply, NumIndicesPerMask]
) -> torch.Tensor:
    """
    Selects features based on batched mask indices.

    Args:
        features: Tensor of shape [B, NumTotalPatches, DimFeatures] containing all patch features.
        batched_mask_indices: Tensor of shape [B, NumMasksToApply, NumIndicesPerMask]
                              containing the indices of patches to select for each mask.
                              Indices should be valid for the NumTotalPatches dimension of features.

    Returns:
        Tensor of shape [B, NumMasksToApply, NumIndicesPerMask, DimFeatures] 
        containing the selected features.
    """
    batch_size, _, dim_features = features.shape
    # Check if batched_mask_indices is empty or has zero size in relevant dimensions
    if batched_mask_indices.nelement() == 0 or batched_mask_indices.shape[1] == 0 or batched_mask_indices.shape[2] == 0:
        # Return an empty tensor with the expected number of dimensions for the output shape if masks are empty
        # Or, decide on specific handling. For now, an empty tensor matching output rank.
        num_masks_to_apply = batched_mask_indices.shape[1] if batched_mask_indices.nelement() > 0 else 0
        num_indices_per_mask = batched_mask_indices.shape[2] if batched_mask_indices.nelement() > 0 else 0
        return torch.empty((batch_size, num_masks_to_apply, num_indices_per_mask, dim_features), 
                           dtype=features.dtype, device=features.device)

    num_masks_to_apply = batched_mask_indices.shape[1]
    num_indices_per_mask = batched_mask_indices.shape[2]

    indices_expanded = batched_mask_indices.unsqueeze(-1).expand(
        batch_size, num_masks_to_apply, num_indices_per_mask, dim_features
    )

    features_expanded = features.unsqueeze(1).expand(
        batch_size, num_masks_to_apply, -1, dim_features 
    )

    selected_features = torch.gather(features_expanded, dim=2, index=indices_expanded)
    
    return selected_features

class IJEPAPredictor(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 predictor_embed_dim: int,
                 depth: int,
                 num_heads: int,
                 num_total_patches: int,
                 # patch_size: int, # Potentially useful for pos_embed details, but num_total_patches more direct for 1D indexing
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.num_total_patches = num_total_patches

        # Positional embeddings for all possible patch locations in the full image
        # These are added to the queries (derived from target patch locations)
        # and potentially to context tokens if they don't already have them.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_total_patches, self.predictor_embed_dim), requires_grad=False)

        # Input projection for context tokens, if backbone_embed_dim != predictor_transformer_embed_dim
        self.context_proj = nn.Linear(self.embed_dim, self.predictor_embed_dim) if self.embed_dim != self.predictor_embed_dim else nn.Identity()

        # Transformer Decoder Layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.predictor_embed_dim,
            nhead=num_heads,
            dim_feedforward=int(self.predictor_embed_dim * mlp_ratio),
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True # Using Pre-LN is common
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

        # Output projection: maps from predictor_embed_dim back to the original backbone feature dimension
        self.out_proj = nn.Linear(self.predictor_embed_dim, self.embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize positional embeddings using a utility (e.g., from solo.utils.misc)
        # Assuming a square grid of patches for generate_2d_sincos_pos_embed
        grid_size = int(self.num_total_patches**0.5)
        if grid_size * grid_size != self.num_total_patches:
            raise ValueError("num_total_patches must be a perfect square for 2D sin-cos positional embeddings.")
        
        # Attempt to import the utility; provide a fallback if not available for standalone testing
        try:
            from solo.utils.misc import generate_2d_sincos_pos_embed
            pos_embed_data = generate_2d_sincos_pos_embed(
                embed_dim=self.pos_embed.shape[-1],
                grid_size=grid_size,
                cls_token=False # No CLS token for patch embeddings here
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float().unsqueeze(0))
        except ImportError:
            print("Warning: solo.utils.misc.generate_2d_sincos_pos_embed not found. Using random positional embeddings for predictor.")
            nn.init.normal_(self.pos_embed, std=0.02)


        # Initialize other weights (Xavier uniform for linear, etc.)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, 
                context_features: torch.Tensor,         # Shape: [B, NumContextPatches, embed_dim]
                batched_masks_pred_indices: torch.Tensor # Shape: [B, NumPredMasks, K_pred_patches_per_mask]
               ):
        """
        Args:
            context_features: Features from the online encoder (context blocks).
                              Assumed to already have appropriate positional information from the encoder.
            batched_masks_pred_indices: 1D indices of the target patches to be predicted.
        """
        batch_size = context_features.shape[0]
        device = context_features.device

        # 1. Project context features to the predictor's internal dimension
        # context_features are [B, NumContextPatches, embed_dim]
        memory = self.context_proj(context_features) # Shape: [B, NumContextPatches, predictor_embed_dim]

        # 2. Create queries for the target patches using their positional embeddings
        # batched_masks_pred_indices: [B, NumPredMasks, K_pred]
        num_pred_masks = batched_masks_pred_indices.shape[1]
        num_patches_per_pred_mask = batched_masks_pred_indices.shape[2]

        # Gather positional embeddings for all target patches
        # self.pos_embed is [1, NumTotalPatches, predictor_embed_dim]
        # We need to expand it to batch size for apply_ijepa_masks
        expanded_pos_embed = self.pos_embed.expand(batch_size, -1, -1)
        
        # apply_ijepa_masks expects [B, Total, Dim] and [B, N_masks, K_indices]
        # It returns [B, N_masks, K_indices, Dim]
        target_queries_with_pos = apply_ijepa_masks(
            expanded_pos_embed, 
            batched_masks_pred_indices
        ) # Shape: [B, NumPredMasks, K_pred, predictor_embed_dim]

        # Reshape queries for TransformerDecoder input: [B, NumPredMasks * K_pred, predictor_embed_dim]
        # This treats all target patches across all masks for a batch item as one sequence.
        queries = target_queries_with_pos.reshape(
            batch_size, 
            num_pred_masks * num_patches_per_pred_mask, 
            self.predictor_embed_dim
        )
        
        # 3. Pass through Transformer Decoder
        # tgt: queries for the decoder - [B, N_target_sequence_len, D_predictor]
        # memory: context from the encoder - [B, N_context_sequence_len, D_predictor]
        # No target masks (tgt_mask) needed as we predict all target patches jointly (not auto-regressively).
        # No memory masks (memory_mask) for now, assuming context_features are already appropriately masked by encoder.
        predicted_latents_flat = self.decoder(queries, memory)
        # Output shape from decoder: [B, NumPredMasks * K_pred, predictor_embed_dim]

        # 4. Project back to the original feature dimension (of the backbone)
        predicted_latents_projected_flat = self.out_proj(predicted_latents_flat)
        # Output shape: [B, NumPredMasks * K_pred, embed_dim]

        # 5. Reshape to match the structure of targets: [B, NumPredMasks, K_pred, embed_dim]
        predicted_latents_structured = predicted_latents_projected_flat.view(
            batch_size, num_pred_masks, num_patches_per_pred_mask, self.embed_dim
        )

        return predicted_latents_structured

class IJEPA(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements I-JEPA (Image-based Joint-Embedding Predictive Architecture).

        Extra cfg settings:
            method_kwargs:
                pred_depth (int): depth of the predictor.
                pred_emb_dim (int): embedding dimension of the predictor.
                # Masking parameters (from MBMaskCollator)
                patch_size (int): patch-size for model training
                enc_mask_scale (Tuple[float, float]): scale of context blocks
                min_keep (int): min number of patches in context block
                pred_mask_scale (Tuple[float, float]): scale of target blocks
                aspect_ratio (Tuple[float, float]): aspect ratio of target blocks
                num_enc_masks (int): number of context blocks
                num_pred_masks (int): number of target blocks
                allow_overlap (bool): whether to allow overlap b/w context and target blocks
        """
        super().__init__(cfg)

        self.pred_depth = cfg.method_kwargs.pred_depth
        self.pred_emb_dim = cfg.method_kwargs.pred_emb_dim
        self.pred_num_heads = cfg.method_kwargs.pred_num_heads # Added new config

        # Masking parameters
        self.patch_size = cfg.method_kwargs.patch_size
        # Assuming backbone is ViT, its patch_embed layer gives num_patches
        # crop_size is also needed for the mask generator.
        self.crop_size = cfg.data.crop_size 

        self.mask_generator = IJEPAMaskGenerator(
            input_size=(self.crop_size, self.crop_size), # from data config
            patch_size=self.patch_size,
            pred_mask_scale=cfg.method_kwargs.pred_mask_scale,
            enc_mask_scale=cfg.method_kwargs.enc_mask_scale,
            aspect_ratio=cfg.method_kwargs.aspect_ratio,
            nenc=cfg.method_kwargs.num_enc_masks,
            npred=cfg.method_kwargs.num_pred_masks,
            allow_overlap=cfg.method_kwargs.allow_overlap,
            min_keep=cfg.method_kwargs.min_keep
        )

        # Backbone (Encoder) - features_dim is from BaseMethod, assumes backbone is set
        # For I-JEPA, the backbone (encoder) takes image + encoder_masks
        # If current self.backbone (e.g. a standard ViT) doesn't support this,
        # we might need to wrap it or modify its forward method.
        # For now, assume self.backbone.forward(x, masks_enc=...) is possible.

        # Predictor
        # The I-JEPA predictor is a Transformer.
        # It takes context embeddings Z and context_masks + target_masks to predict target_embeddings.
        # Input to predictor: self.features_dim (output of backbone)
        # Output of predictor: self.pred_emb_dim (usually same as features_dim for JEPA)
        # This needs to be a proper Transformer decoder like in Meta I-JEPA.
        # For now, a simple MLP as placeholder, will need to be replaced.
        self.predictor = IJEPAPredictor(
            embed_dim=self.features_dim, 
            predictor_embed_dim=self.pred_emb_dim,
            depth=self.pred_depth,
            num_heads=self.pred_num_heads,
            num_total_patches=(self.crop_size // self.patch_size)**2 
        )
        print(f"Warning: IJEPA Predictor is a placeholder MLP, not a Transformer.")


        # Momentum backbone (target_encoder) - already created by BaseMomentumMethod
        # self.momentum_backbone

        # No separate momentum predictor for I-JEPA normally.
        # The target is generated by the momentum_backbone.

        # Initialize momentum params (already done for backbone in BaseMomentumMethod)
        # We don't have a momentum_predictor in typical I-JEPA.

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config."""
        cfg = super(IJEPA, IJEPA).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_depth")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_emb_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.patch_size")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_num_heads") # Added
        # Add defaults for other I-JEPA masking params if desired, or assert them
        # For example:
        cfg.method_kwargs.enc_mask_scale = cfg.method_kwargs.get("enc_mask_scale", (0.85, 1.0))
        cfg.method_kwargs.pred_mask_scale = cfg.method_kwargs.get("pred_mask_scale", (0.15, 0.2))
        cfg.method_kwargs.aspect_ratio = cfg.method_kwargs.get("aspect_ratio", (0.75, 1.5))
        cfg.method_kwargs.num_enc_masks = cfg.method_kwargs.get("num_enc_masks", 1)
        cfg.method_kwargs.num_pred_masks = cfg.method_kwargs.get("num_pred_masks", 4)
        cfg.method_kwargs.allow_overlap = cfg.method_kwargs.get("allow_overlap", False)
        cfg.method_kwargs.min_keep = cfg.method_kwargs.get("min_keep", 10) # Min patches per mask

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds predictor parameters to the parent's learnable parameters."""
        extra_learnable_params = [
            {"name": "predictor", "params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    # No separate momentum_pairs for predictor needed if it's not a momentumized component

    def forward_features(
        self, x: torch.Tensor, masks_enc: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Helper function to get features from the online backbone using encoder masks.
        This might need adaptation depending on how the backbone handles masks.
        """
        # This assumes self.backbone.forward can take masks_enc.
        # If not, we apply masks *after* getting full features.
        # For a ViT, masks_enc would typically be a list of tensors,
        # each tensor containing indices of patches to *keep* for that mask.
        # The backbone should then process only these patches.
        
        # Option 1: Backbone handles masking internally (ideal)
        # feats = self.backbone(x, masks_enc=masks_enc) 
        
        # Option 2: Get full features, then apply masks (less efficient for ViT encoder)
        # This is not how original I-JEPA ViT encoder works.
        # It directly processes only the visible patches defined by masks_enc.
        # We need to ensure our backbone (likely a ViT) can do this.
        # Most ViT implementations in solo-learn expect full images.
        
        # For now, let's assume a placeholder that we'll refine.
        # This part is CRITICAL and depends on the backbone's capabilities.
        
        # If backbone is a ViT, we expect patch embeddings.
        # Example: ViT forward might look like:
        # x = self.backbone.patch_embed(x)
        # x = self.backbone.pos_drop(x)
        # # Apply masks_enc here, selecting only a subset of patch tokens
        # # Then proceed with Transformer blocks on these selected tokens.
        
        # This will require adapting the ViT forward pass.
        # For now, let's return full features and assume masking is handled by predictor input preparation.
        # This is NOT how I-JEPA does it for the encoder, but a starting point.
        if hasattr(self.backbone, 'forward_features_masked') and masks_enc is not None:
             # Ideal scenario if we modify ViT to have this
            feats = self.backbone.forward_features_masked(x, masks_enc)
        else:
            # Fallback: Get all features, predictor will have to work with this + masks
            print("Warning: IJEPA online encoder is not using masks_enc directly for feature extraction. Full features are returned.")
            feats = self.backbone(x) # This is likely [B, N_total_patches, D] or [B, D] if pooled
            # If feats is [B,D] (global), this whole masking strategy breaks down for the encoder.
            # We absolutely need patch-level features from the backbone.
            # Solo-learn backbones usually return [B, D] from self.backbone(x).
            # We need self.backbone.forward_features(x) -> [B, NumPatches, Dim]
            if feats.ndim == 2: # Likely a pooled global feature
                raise ValueError("IJEPA requires patch-level features from the backbone's `forward_features` method, not global features.")

        return feats

    def _collate_masks(self, 
                       list_of_mask_lists: List[List[torch.Tensor]], 
                       num_masks_per_item: int, # N_enc or N_pred
                       device: torch.device
                      ) -> torch.Tensor:
        """
        Collates a list of lists of mask tensors into a single batch tensor.
        Each mask tensor contains indices. All masks in the batch will be truncated
        to the length of the shortest mask found across the entire batch, or to
        self.mask_generator.min_keep if all are longer.

        Args:
            list_of_mask_lists: List (batch_size) of lists (num_masks_per_item) of tensors (indices).
            num_masks_per_item: The expected number of masks per batch item (e.g., self.mask_generator.nenc).
            device: The torch device to place the collated tensor on.

        Returns:
            A tensor of shape [B, num_masks_per_item, K_fixed_indices_per_mask].
        """
        batch_size = len(list_of_mask_lists)
        if batch_size == 0:
            return torch.empty((0, num_masks_per_item, 0), dtype=torch.long, device=device)

        min_len_k = float('inf')
        found_any_mask = False
        for i in range(batch_size):
            if len(list_of_mask_lists[i]) != num_masks_per_item:
                # This case should ideally not happen if mask_generator is consistent
                # Or we need to pad/truncate the number of masks per item as well.
                # For now, assume consistency or raise error.
                raise ValueError(f"Inconsistent number of masks for batch item {i}. Expected {num_masks_per_item}, got {len(list_of_mask_lists[i])}")
            for mask_tensor in list_of_mask_lists[i]:
                if mask_tensor.numel() > 0:
                    min_len_k = min(min_len_k, mask_tensor.numel())
                    found_any_mask = True
        
        if not found_any_mask:
             # All masks in the batch are empty, K_fixed will be 0
            k_fixed = 0
        elif min_len_k == float('inf'): # Should be caught by found_any_mask, but as a safeguard
            k_fixed = 0
        else:
            # Ensure k_fixed is at least min_keep, but not more than the shortest actual mask length found
            # If the shortest mask is already < min_keep, IJEPAMaskGenerator might have issues or returned empty.
            # Here we trust min_len_k reflects a valid mask length from generator.
            k_fixed = max(self.mask_generator.min_keep, min_len_k) 
            # If all masks are very long, but we only need min_keep, we can cap at min_keep.
            # However, the original IJEPA takes the shortest in batch to ensure no OOB if some masks are short.
            # So, min_len_k (actual shortest) is a better primary determinant if it's >= min_keep.
            # If min_len_k from data is < self.mask_generator.min_keep, it implies an issue upstream or empty masks.
            # Let's stick to min_len_k if it was found, otherwise default to 0 if no masks. Or enforce min_keep.
            if min_len_k < self.mask_generator.min_keep and found_any_mask:
                # This scenario implies that the shortest non-empty mask is smaller than min_keep.
                # This might be okay if min_keep in generator is a soft target for sampling attempts.
                # Let's use the actual shortest length found to avoid errors, 
                # but it might be very small. The loss fn or predictor might need to handle this.
                pass # min_len_k is already set
            k_fixed = min_len_k # Use the true shortest length found among non-empty masks.

        # If k_fixed ended up as infinity (no masks found), set to 0
        if k_fixed == float('inf'):
            k_fixed = 0

        collated_tensor = torch.zeros((batch_size, num_masks_per_item, k_fixed), dtype=torch.long, device=device)

        for i in range(batch_size):
            for j in range(num_masks_per_item):
                mask_tensor = list_of_mask_lists[i][j]
                if mask_tensor.numel() > 0:
                    len_to_take = min(mask_tensor.numel(), k_fixed)
                    collated_tensor[i, j, :len_to_take] = mask_tensor[:len_to_take]
                # If mask_tensor is empty or shorter than k_fixed, the rest of collated_tensor[i,j,:] remains 0.
                # This might require careful handling if 0 is a valid patch index.
                # Using a special padding index (e.g., -1) might be safer if 0 is a valid index.
                # For now, let's assume 0 padding is acceptable or masks are never truly empty post-generation if valid.

        return collated_tensor

    def forward(self, X: torch.Tensor, masks_enc: List[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Forward pass of the online backbone and predictor.
        X: A batch of images.
        masks_enc: Precomputed encoder masks for X. If None, they should be generated.
        """
        if masks_enc is None:
            # This is tricky. Mask generator was designed for collate_fn.
            # For a single batch, we need to generate masks for each image.
            # This is a placeholder, proper batch-wise mask generation is needed.
            batch_masks_enc = []
            batch_masks_pred_unused = [] # We only need enc masks for the online branch typically
            for _ in range(X.shape[0]):
                m_e, _ = self.mask_generator(X.device)
                batch_masks_enc.append(m_e)
            # `batch_masks_enc` is now List[List[torch.Tensor]] (BatchSize x NumEncMasksPerImage)
            # The `forward_features` and `predictor` need to handle this batched list of masks.
            # This part needs careful alignment with how the Meta I-JEPA handles batched masks.
            # For now, let's assume masks_enc for forward_features needs to be a list of tensors,
            # potentially one mask that is the union of all num_enc_masks.
            # The Meta I-JEPA might pass each enc mask and get separate features.
            # For simplicity in this draft, let's assume `masks_enc` is a single representative mask.
            # THIS IS A MAJOR SIMPLIFICATION FOR THE SKELETON.
            current_masks_enc_for_batch = batch_masks_enc # Placeholder
        else:
            current_masks_enc_for_batch = masks_enc

        # Get features from the context blocks (visible to online encoder)
        # This needs to be [B, NumContextPatches, Dim]
        # This is where `self.backbone(X, current_masks_enc_for_batch)` should ideally work.
        # Let's assume forward_features handles this and returns appropriate context features.
        # The original IJEPA: z = encoder(imgs, masks_enc)
        # Here `masks_enc` itself might be List[List[torch.Tensor]] (batch of list of mask indices)
        context_features = self.forward_features(X, current_masks_enc_for_batch)

        out = {"context_features": context_features}
        
        # Predictor is called in training_step using context_features and target_masks.
        # So, this forward primarily provides the context_features.
        return out

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor, masks_pred: List[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Forward pass of the momentum backbone to get target latents.
        X: A batch of images.
        masks_pred: Precomputed predictor masks (target masks) for X.
        """
        if masks_pred is None:
            batch_masks_pred = []
            for _ in range(X.shape[0]):
                _, m_p = self.mask_generator(X.device)
                batch_masks_pred.append(m_p)
            current_masks_pred_for_batch = batch_masks_pred # Placeholder for batched masks
        else:
            current_masks_pred_for_batch = masks_pred

        # Get full features from momentum backbone
        # Assuming self.momentum_backbone.forward_features(X) returns [B, NumTotalPatches, Dim]
        if hasattr(self.momentum_backbone, 'forward_features'):
            h_full = self.momentum_backbone.forward_features(X)
        else: # Fallback if only standard forward is available, hoping it returns patch features
            h_full = self.momentum_backbone(X) 
            if h_full.ndim == 2:
                 raise ValueError("IJEPA momentum_backbone must provide patch-level features.")


        h_full_norm = F.layer_norm(h_full, (h_full.size(-1),))

        # Apply predictor masks to get target latents
        # `current_masks_pred_for_batch` is List[List[torch.Tensor]] (BatchSize x NumPredMasksPerImage)
        # `apply_ijepa_masks` needs to handle this.
        # Result should be target latents for each prediction mask.
        # Shape: e.g., [B, NumPredMasks, NumPatchesInPredMask, Dim]
        target_latents = apply_ijepa_masks(h_full_norm, current_masks_pred_for_batch)
        
        return {"k": target_latents}


    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for I-JEPA."""
        # BaseMomentumMethod already handles momentum updates for backbone
        super().training_step(batch, batch_idx) 

        X, _ = batch # Assuming batch is (images, labels/targets)
        X = X[0] # Taking the first view if multiple views are present (like in other solo-learn methods)
        
        # 1. Generate masks for the current batch X
        # This is a simplification. MBMaskCollator is complex.
        # We need batched masks: masks_enc_batch [B, N_enc, K_enc], masks_pred_batch [B, N_pred, K_pred]
        # For now, mask_generator called per image in forward/momentum_forward is a stand-in.
        # Let's assume forward() and momentum_forward() can be called with X and will internally handle masks.
        
        # This is the most complex part to adapt from Meta I-JEPA:
        # The predictor takes `z` (from online encoder + enc_masks) 
        # AND `masks_enc` AND `masks_pred` to generate predictions for `masks_pred` locations.
        # z = encoder(imgs, masks_enc) -> [B, N_enc_total_patches, D] or [B*N_enc, K_enc, D]
        # p = predictor(z, masks_enc, masks_pred) -> [B, N_pred_total_patches, D] or [B*N_pred, K_pred, D]
        
        # Simplified flow for now:
        # a. Get context features using online encoder and encoder masks
        # This needs `masks_enc` to be generated for the batch `X`.
        # Let's call our forward, assuming it prepares context_features
        # The placeholder `IJEPAMaskGenerator` and `apply_ijepa_masks` are critical.
        
        # To properly implement, we need masks for the whole batch first.
        batch_size = X.shape[0]
        device = X.device
        
        list_masks_enc_for_batch = []
        list_masks_pred_for_batch = []
        for _ in range(batch_size):
            m_e, m_p = self.mask_generator(device=device) # m_e is List[Tensor], m_p is List[Tensor]
            list_masks_enc_for_batch.append(m_e)
            list_masks_pred_for_batch.append(m_p)
        
        # We need to collate these lists of lists into batch tensors for masks if predictor/apply_masks expect that.
        # Original MBMaskCollator produces:
        # masks_enc: [B, N_enc, K_enc_max] (tensor)
        # masks_pred: [B, N_pred, K_pred_max] (tensor)
        # This requires padding or careful handling if K varies.
        # For now, our dummy apply_ijepa_masks takes List[List[torch.Tensor]] via current_masks_pred_for_batch in momentum_forward.
        # Let's assume a similar structure for the predictor call.
        
        # --- Online branch: Get context features ---
        # The `self.backbone` in solo-learn might need a wrapper or direct modification
        # to handle `masks_enc` and provide features only for those context blocks.
        # E.g. `context_features = self.backbone(X, list_masks_enc_for_batch)`
        # This is a placeholder, true I-JEPA ViT encoder takes tokens corresponding to masks_enc.
        print("Warning: IJEPA training_step needs proper masked backbone forward.")
        full_online_features = self.backbone.forward_features(X) # [B, N_total, D]
        
        # For now, assume `context_features` are extracted by some mechanism based on `list_masks_enc_for_batch`.
        # This is where the true I-JEPA `encoder(imgs, masks_enc)` call happens.
        # It should result in something like [B, N_enc_masks * K_enc_patches, D]
        # For simplicity, let's pass full features and masks to predictor.
        context_features_for_predictor = full_online_features # This is NOT how I-JEPA predictor gets its input normally.
                                                            # Predictor needs features *from the context blocks only*.

        # --- Target branch: Get target latents ---
        # Assuming momentum_forward handles masks_pred correctly.
        # It should return a dict {"k": target_latents}
        # where target_latents could be [B, N_pred, K_pred, D]
        target_output = self.momentum_forward(X, list_masks_pred_for_batch)
        h_target_latents = target_output["k"] # Shape e.g. [B, N_pred, K_pred, D]

        # --- Predictor step ---
        # The predictor needs context_features, list_masks_enc_for_batch, and list_masks_pred_for_batch
        # Its output should match the shape of h_target_latents.
        # `z_predicted_latents = self.predictor(context_features_for_predictor, list_masks_enc_for_batch, list_masks_pred_for_batch)`
        # The placeholder predictor is just an MLP. A real I-JEPA predictor is a Transformer
        # that takes positional information from masks.
        print(f"Warning: IJEPA predictor call is a placeholder. Input shape: {context_features_for_predictor.shape}")
        # This MLP predictor cannot use the masks effectively.
        # We'd need to reshape/select features for the MLP.
        # For a true Transformer predictor:
        # It would take `z = online_encoder(X, masks_enc)` which are features of context blocks.
        # Then `p = predictor(z, masks_enc_pos, masks_pred_pos)`
        
        # Gross simplification for placeholder MLP predictor:
        # Reshape context_features_for_predictor if it's [B, N, D] to something the MLP can take.
        # Reshape target_latents if it's [B, M, K, D] to [B, M*K, D] then to [B, D_mlp_target]
        # This is just to make the loss runnable with the MLP.
        
        # Assume predictor outputs something compatible with a reshaped h_target_latents.
        # This will need significant work to align with actual I-JEPA predictor.
        # For now, let's imagine the predictor somehow outputs a tensor that can be compared
        # to a flattened version of h_target_latents.
        # If h_target_latents is [B, N_pred, K_pred, D], its total elements per batch item are N_pred*K_pred*D
        # Predictor input (simplified): global context features.
        # This prediction step is the most hand-wavy part right now.
        
        # Let's simulate a prediction that matches the target latent dimension for the MLP.
        # The MLP takes [B, D_features] and outputs [B, D_features].
        # We need to make h_target_latents look like [B, D_features] for comparison.
        # This is NOT how I-JEPA works but makes the placeholder runnable.
        avg_context_features = context_features_for_predictor.mean(dim=1) # [B, D]
        z_predicted_for_loss = self.predictor(avg_context_features) # [B, D]

        # Make target compatible for the placeholder MLP loss
        # h_target_latents is e.g. [B, N_pred, K_pred, D]
        # Take mean over N_pred and K_pred to get [B, D]
        if h_target_latents.ndim == 4:
            h_target_for_loss = h_target_latents.mean(dim=(1,2)) # [B, D]
        else: # If it was already [B, Something, D]
            h_target_for_loss = h_target_latents.mean(dim=1) if h_target_latents.ndim == 3 else h_target_latents

        loss = F.smooth_l1_loss(z_predicted_for_loss, h_target_for_loss.detach())

        # Log I-JEPA specific metrics
        self.log_dict({"train_ijepa_loss": loss}, on_epoch=True, on_step=True, sync_dist=True)
        return loss 