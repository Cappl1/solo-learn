import random
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
from solo.losses.mocov3 import mocov3_loss_func
from solo.methods.curriculum_mocov3 import CurriculumMoCoV3
from solo.utils.misc import gather


from solo.methods.curriculum_mocov3 import CurriculumMoCoV3  # noqa: your base
from solo.losses.mocov3 import mocov3_loss_func


class SelectiveJEPACurriculumMoCoV3(CurriculumMoCoV3):
    """
    MoCo v3 → curriculum via JEPA-based difficulty and hard selection.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        kw = cfg.method_kwargs
        self.num_candidates = kw.get("num_candidates", 8)
        self.curriculum_warmup_epochs = kw.get("curriculum_warmup_epochs", 20)
        self.curriculum_reverse = kw.get("curriculum_reverse", False)
        self.selection_epochs = kw.get("selection_epochs", 100)  # renamed
        self.register_buffer("_global_step", torch.tensor(0))

    # helper -----------------------------------------------------------------
    @staticmethod
    def _batched_errors(
        backbone, projector, key_feats: torch.Tensor, cand_imgs: torch.Tensor, B: int, K: int
    ) -> torch.Tensor:
        """
        Compute reconstruction MSE for all B×K pairs in *one* no-grad pass.
        Returns shape (B,K).
        """
        with torch.no_grad():
            cand_feats = SelectiveJEPACurriculumMoCoV3._feats(backbone(cand_imgs)).view(B, K, -1)  # (B,K,D)
            pred_feats = projector(key_feats).unsqueeze(1).expand_as(cand_feats)
            errs = F.mse_loss(pred_feats, cand_feats, reduction="none").mean(-1)  # (B,K)
        return errs  # no graph

    # ------------------------------------------------------------------ #
    @staticmethod
    def _feats(backbone_out):
        """Return a 2-D feature tensor regardless of backbone output type."""
        if isinstance(backbone_out, dict):
            # common keys in solo-learn are 'feats' or 'logits'
            return backbone_out.get("feats") or backbone_out[list(backbone_out.keys())[0]]
        return backbone_out            # already a tensor

    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        # ---- unpack ------------------------------------------------------
        idxs, keyframes, cand_tensor, _ = batch         # (B) (B,C,H,W) (B,K,C,H,W)
        keyframes   = keyframes.to(self.device, non_blocking=True)
        cand_tensor = cand_tensor.to(self.device, non_blocking=True)
        B, K = cand_tensor.shape[:2]

        # ---------- curriculum selection ---------------------------------
        use_selection = self.current_epoch < self.selection_epochs
        
        # Calculate progress through curriculum warmup (for logging)
        progress = (
            min(1.0, self.current_epoch / self.curriculum_warmup_epochs)
            if self.curriculum_warmup_epochs > 0
            else 1.0
        )
        
        if use_selection:
            key_feats = self._feats(self.backbone(keyframes))               # (B,D)
            errs = self._batched_errors(
                self.backbone,
                self.predictor_jepa,
                key_feats,
                cand_tensor.view(B * K, *cand_tensor.shape[2:]),
                B,
                K,
            )                                                             # (B,K)

            # Log error statistics from selection
            self.log("curriculum_errors_min", errs.min(), on_step=False, on_epoch=True)
            self.log("curriculum_errors_max", errs.max(), on_step=False, on_epoch=True)
            self.log("curriculum_errors_median", errs.median(), on_step=False, on_epoch=True)
            self.log("curriculum_errors_mean", errs.mean(), on_step=False, on_epoch=True)
            self.log("curriculum_errors_std", errs.std(), on_step=False, on_epoch=True)
            
            sel_idx = []
            for b in range(B):
                # Always choose the hardest (max error) or easiest (min error) sample
                # based on curriculum_reverse, without progressive transition
                if self.curriculum_reverse:
                    # Select hardest sample (highest error)
                    idx = torch.argmax(errs[b]).item()
                else:
                    # Select easiest sample (lowest error)
                    idx = torch.argmin(errs[b]).item()
                sel_idx.append(idx)
            sel_idx = torch.tensor(sel_idx, device=self.device)

            # In selective approach, we select based on difficulty setting
            selection_type = "hardest" if self.curriculum_reverse else "easiest"
            
            # Log position metric - now based on selection type
            self.log("curriculum_selection_type", 1.0 if self.curriculum_reverse else 0.0, on_step=True, on_epoch=True)
            
            # gather chosen candidates
            x2 = cand_tensor[torch.arange(B, device=self.device), sel_idx]  # (B,C,H,W)
            selected_errors = errs[torch.arange(B, device=self.device), sel_idx]
            log_err = selected_errors.mean().item()
            
            # Log the distribution of selected sample errors
            self.log("curriculum_selected_error_mean", selected_errors.mean(), on_step=True, on_epoch=True) 
            self.log("curriculum_selected_error_min", selected_errors.min(), on_step=False, on_epoch=True)
            self.log("curriculum_selected_error_max", selected_errors.max(), on_step=False, on_epoch=True)
            self.log("curriculum_selected_error_std", selected_errors.std(), on_step=False, on_epoch=True)
        else:
            rand_idx = torch.randint(0, K, (B,), device=self.device)
            x2 = cand_tensor[torch.arange(B, device=self.device), rand_idx]
            log_err = 0.0
            
            # Log placeholder values for consistency when not using selection
            self.log("curriculum_selection_type", -1.0, on_step=True, on_epoch=True)  # -1 indicates random selection
            self.log("curriculum_selected_error_mean", 0.0, on_step=True, on_epoch=True)

        x1 = keyframes

        # ---------- MoCo forward / loss ----------------------------------
        out1 = self.forward(x1)
        out2 = self.forward(x2)
        m_out1 = self.momentum_forward(x1)
        m_out2 = self.momentum_forward(x2)

        q1, q2 = out1["q"], out2["q"]
        k1, k2 = m_out1["k"].detach(), m_out2["k"].detach()

        contrastive_loss = 0.5 * (
            mocov3_loss_func(q1, k2, temperature=self.temperature)
            + mocov3_loss_func(q2, k1, temperature=self.temperature)
        )

        # ---------- projector update (no backbone grad) ------------------
        with torch.no_grad():
            feat1 = out1["feats"].detach()
            feat2 = out2["feats"].detach()

        recon_pred = self.predictor_jepa(feat1)
        recon_loss = F.mse_loss(recon_pred, feat2)
        
        # Calculate per-sample reconstruction errors for logging
        per_sample_error = F.mse_loss(recon_pred, feat2, reduction='none').mean(dim=1)
        
        # Log statistics of per-sample reconstruction errors
        self.log("train_reconstruction_err_mean", per_sample_error.mean(), on_step=True, on_epoch=True)
        self.log("train_reconstruction_err_min", per_sample_error.min(), on_step=False, on_epoch=True)
        self.log("train_reconstruction_err_max", per_sample_error.max(), on_step=False, on_epoch=True)

        total_loss = contrastive_loss + recon_loss

        # Log comprehensive metrics matching the parent class format
        self.log_dict(
            {
                # Use parent class naming convention for losses
                "train_contrastive_loss": contrastive_loss,
                "train_reconstruction_loss": recon_loss,
                "train_total_loss": total_loss,
                
                # Error metrics - keep the original naming for backward compatibility
                "train/sel_error": log_err,
                
                # Selection and curriculum metrics - updated to reflect the new approach
                "train/use_selection": float(use_selection),
                "curriculum_phase": self.current_epoch / max(1, self.curriculum_warmup_epochs),
                "curriculum_progress": progress,
                "curriculum_direction": 1.0 if self.curriculum_reverse else 0.0,
                
                # Selection strategy metrics
                "curriculum_selection_active": float(use_selection),
                "curriculum_selection_epoch_ratio": self.current_epoch / max(1, self.selection_epochs),
            },
            on_step=True,
            on_epoch=True,
            sync_dist=True,  # Important for distributed training
            prog_bar=True,   # Keep the progress bar display
        )

        # Increment step counter
        self._global_step += 1

        return total_loss