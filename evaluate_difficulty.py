"""
Standalone script to evaluate sample difficulty and classification accuracy.
This can be run after training to analyze the relationship between difficulty and performance.

Usage:
python evaluate_difficulty.py \
    --model_path /path/to/curriculum_mocov3.ckpt \
    --linear_path /path/to/linear_classifier.ckpt \
    --cfg_path /path/to/config.yaml \
    --difficulty_method reconstruction \
    --batch_size 64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
import pandas as pd
import warnings
import random
warnings.filterwarnings("ignore")

from omegaconf import OmegaConf
from solo.methods.curriculum_mocov3 import CurriculumMoCoV3
from solo.data.classification_dataloader import prepare_data


class DifficultyEvaluator:
    """Evaluates sample difficulty and classification performance using actual temporal pairs."""
    
    def __init__(self, model_path: str, linear_path: str, cfg_path: str, 
                 device: str = 'cuda', difficulty_method: str = 'reconstruction'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.difficulty_method = difficulty_method
        
        print(f"Loading configuration from {cfg_path}")
        # Load config - handle dummy path case
        if cfg_path != "dummy":
            self.cfg = OmegaConf.load(cfg_path)
        else:
            # Config will be set later in main function
            self.cfg = None
        
        print(f"Loading pretrained model from {model_path}")
        # Load pretrained model (CurriculumMoCoV3) - defer if config not ready
        if self.cfg is not None:
            self.model = self.load_pretrained_model(model_path)
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model_path = model_path  # Store for later loading
        
        print(f"Loading linear classifier from {linear_path}")
        # Load linear classifier
        self.classifier = self.load_linear_classifier(linear_path)
        self.classifier.to(self.device)
        self.classifier.eval()
        
        # Storage for results
        self.results = {
            'difficulties': [],
            'predictions': [],
            'targets': [],
            'correct': [],
            'indices': [],
            'features': []  # Optional: store features for analysis
        }
        
        print(f"Evaluator initialized with {difficulty_method} difficulty method")
    
    def finalize_model_loading(self):
        """Complete model loading after config is set."""
        if hasattr(self, 'model_path') and self.cfg is not None:
            print(f"Finalizing model loading from {self.model_path}")
            self.model = self.load_pretrained_model(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            delattr(self, 'model_path')  # Clean up
    
    def load_pretrained_model(self, model_path):
        """Load the pretrained CurriculumMoCoV3 model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # Check if curriculum parameters are missing in config and add defaults
        if 'method_kwargs' not in self.cfg:
            self.cfg.method_kwargs = {}
        
        # Check if data section is missing and add defaults
        if 'data' not in self.cfg:
            self.cfg.data = {}
        
        # Add missing curriculum parameters with reasonable defaults
        curriculum_defaults = {
            'curriculum_type': 'mae',
            'curriculum_strategy': 'exponential', 
            'curriculum_warmup_epochs': 10,
            'curriculum_weight': 1.0,
            'reconstruction_masking_ratio': 0.75,
            'curriculum_reverse': False,
            'proj_hidden_dim': 4096,
            'proj_output_dim': 256,
            'pred_hidden_dim': 4096,
            'temperature': 0.1
        }
        
        # Add missing data parameters with reasonable defaults
        data_defaults = {
            'num_classes': 50,  # Default for Core50 instance-level
            'dataset': 'core50',
            'format': 'h5',
            'num_workers': 4,
            'num_large_crops': 2,
            'num_small_crops': 0,
            'fraction': 1.0
        }
        
        # Add missing general config parameters
        general_defaults = {
            'no_validation': False,
            'max_epochs': 100,
            'devices': [0],
            'accelerator': 'gpu',
            'strategy': 'ddp',
            'precision': 32
        }
        
        for key, default_value in curriculum_defaults.items():
            if key not in self.cfg.method_kwargs:
                self.cfg.method_kwargs[key] = default_value
                print(f"Added missing parameter {key} = {default_value}")
                
        for key, default_value in data_defaults.items():
            if key not in self.cfg.data:
                self.cfg.data[key] = default_value
                print(f"Added missing data parameter {key} = {default_value}")
        
        for key, default_value in general_defaults.items():
            if key not in self.cfg:
                self.cfg[key] = default_value
                print(f"Added missing general parameter {key} = {default_value}")
        
        # Try to infer curriculum_type from checkpoint keys
        has_decoder = any('decoder' in k for k in state_dict.keys())
        has_predictor_jepa = any('predictor_jepa' in k for k in state_dict.keys())
        
        if has_decoder:
            self.cfg.method_kwargs.curriculum_type = 'mae'
            print("Detected MAE curriculum type from checkpoint (has decoder)")
        elif has_predictor_jepa:
            self.cfg.method_kwargs.curriculum_type = 'jepa'
            print("Detected JEPA curriculum type from checkpoint (has predictor_jepa)")
        
        # Initialize model
        try:
            model = CurriculumMoCoV3(self.cfg)
        except Exception as e:
            print(f"Error initializing CurriculumMoCoV3: {e}")
            print("Falling back to basic parameters...")
            
            # Fallback: ensure we have the minimum required parameters
            required_params = ['proj_hidden_dim', 'proj_output_dim', 'pred_hidden_dim', 'temperature']
            for param in required_params:
                if param not in self.cfg.method_kwargs:
                    if param.endswith('_dim'):
                        self.cfg.method_kwargs[param] = 4096 if 'hidden' in param else 256
                    else:
                        self.cfg.method_kwargs[param] = 0.1
            
            # Ensure data section has all required fields
            required_data_params = ['num_classes', 'dataset', 'format', 'num_workers', 'num_large_crops', 'num_small_crops', 'fraction']
            for param in required_data_params:
                if param not in self.cfg.data:
                    if param == 'num_classes':
                        self.cfg.data[param] = 50
                    elif param == 'dataset':
                        self.cfg.data[param] = 'core50'
                    elif param == 'format':
                        self.cfg.data[param] = 'h5'
                    elif param == 'num_workers':
                        self.cfg.data[param] = 4
                    elif param == 'num_large_crops':
                        self.cfg.data[param] = 2
                    elif param == 'num_small_crops':
                        self.cfg.data[param] = 0
                    elif param == 'fraction':
                        self.cfg.data[param] = 1.0
            
            # Ensure general config has all required fields
            required_general_params = ['no_validation', 'max_epochs', 'devices', 'accelerator', 'strategy', 'precision']
            for param in required_general_params:
                if param not in self.cfg:
                    if param == 'no_validation':
                        self.cfg[param] = False
                    elif param == 'max_epochs':
                        self.cfg[param] = 100
                    elif param == 'devices':
                        self.cfg[param] = [0]
                    elif param == 'accelerator':
                        self.cfg[param] = 'gpu'
                    elif param == 'strategy':
                        self.cfg[param] = 'ddp'
                    elif param == 'precision':
                        self.cfg[param] = 32
            
            model = CurriculumMoCoV3(self.cfg)
        
        # Load state dict, ignoring non-matching keys
        # Filter out classifier weights since we're loading our own separate classifier
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.')}
        
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys in checkpoint: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        
        print(f"Loaded CurriculumMoCoV3 model: {model.backbone_name}, curriculum_type: {model.curriculum_type}")
        
        return model
    
    def load_linear_classifier(self, linear_path):
        """Load the trained linear classifier."""
        checkpoint = torch.load(linear_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # Find classifier weights in the state dict
        classifier_weight = None
        classifier_bias = None
        
        for key in state_dict.keys():
            if 'classifier.weight' in key:
                classifier_weight = state_dict[key]
            elif 'classifier.bias' in key:
                classifier_bias = state_dict[key]
        
        if classifier_weight is None:
            raise ValueError("Could not find classifier weights in checkpoint")
        
        # Extract dimensions
        num_classes, num_features = classifier_weight.shape
        
        print(f"Linear classifier: {num_features} features -> {num_classes} classes")
        print(f"Classifier weight shape: {classifier_weight.shape}")
        if classifier_bias is not None:
            print(f"Classifier bias shape: {classifier_bias.shape}")
        
        # Create classifier
        classifier = nn.Linear(num_features, num_classes)
        classifier.weight.data = classifier_weight
        if classifier_bias is not None:
            classifier.bias.data = classifier_bias
        
        return classifier
    
    def compute_temporal_reconstruction_difficulty(self, img1, img2):
        """Compute difficulty based on predicting img1 (classification target) from img2 (temporal context)."""
        with torch.no_grad():
            try:
                # Get features for both temporal frames
                # img1 = classification target, img2 = temporal context
                out1 = self.model.forward(img1)
                out2 = self.model.forward(img2)
                
                # Handle different backbone types
                if self.model.is_vit:
                    # ViT backbone
                    feats1 = out1.get('feats_vit', out1.get('feats'))  # Target features
                    feats2 = out2.get('feats_vit', out2.get('feats'))  # Context features
                    
                    if self.model.curriculum_type == "mae":
                        # Check if the model has the required decoder
                        if not hasattr(self.model, 'decoder'):
                            print("Warning: Model doesn't have decoder, falling back to dummy reconstruction difficulty")
                            return torch.rand(img1.shape[0])
                        
                        # MAE-style: reconstruct img1 patches from img2 features
                        B = feats2.shape[0]
                        num_tokens = feats2.shape[1]
                        mask_ratio = self.model.masking_ratio
                        
                        # Create mask for each sample
                        mask_start_idx = 1 if hasattr(self.model.backbone, "cls_token") else 0
                        mask = torch.zeros(B, num_tokens, dtype=torch.bool, device=feats2.device)
                        mask_length = int(mask_ratio * (num_tokens - mask_start_idx))
                        
                        for i in range(B):
                            mask_indices = torch.randperm(num_tokens - mask_start_idx, device=feats2.device)[:mask_length]
                            mask[i, mask_indices + mask_start_idx] = True
                        
                        # Predict img1 features from img2 features
                        _, per_sample_error = self._compute_mae_reconstruction_temporal(feats2, feats1, mask, None)
                        
                    else:  # JEPA
                        # Check if the model has the required predictor
                        if not hasattr(self.model, 'predictor_jepa'):
                            print("Warning: Model doesn't have predictor_jepa, falling back to dummy reconstruction difficulty")
                            return torch.rand(img1.shape[0])
                        
                        # JEPA-style: predict img1 features from img2 features
                        B = feats2.shape[0]
                        num_tokens = feats2.shape[1]
                        
                        # Use temporal relationship for context/target split
                        target_ratio = self.model.masking_ratio
                        split_start_idx = 1 if hasattr(self.model.backbone, "cls_token") else 0
                        target_length = int(target_ratio * (num_tokens - split_start_idx))
                        
                        context_mask = torch.ones(B, num_tokens, dtype=torch.bool, device=feats2.device)
                        target_mask = torch.zeros(B, num_tokens, dtype=torch.bool, device=feats2.device)
                        
                        for i in range(B):
                            indices_perm = torch.randperm(num_tokens - split_start_idx, device=feats2.device)
                            target_indices = indices_perm[:target_length] + split_start_idx
                            context_mask[i, target_indices] = False
                            target_mask[i, target_indices] = True
                        
                        # Predict img1 features from img2 context
                        _, per_sample_error = self._compute_jepa_reconstruction_temporal(feats2, feats1, context_mask, target_mask)
                        
                else:
                    # ResNet/CNN backbone
                    feats1 = out1.get('feats')  # Target features (img1)
                    feats2 = out2.get('feats')  # Context features (img2)
                    
                    if feats1 is None:
                        feats1 = out1 if isinstance(out1, torch.Tensor) else out1.get('z', out1)
                    if feats2 is None:
                        feats2 = out2 if isinstance(out2, torch.Tensor) else out2.get('z', out2)
                    
                    if self.model.curriculum_type == "mae":
                        # Check if the model has the required decoder
                        if not hasattr(self.model, 'decoder'):
                            print("Warning: Model doesn't have decoder, falling back to dummy reconstruction difficulty")
                            return torch.rand(img1.shape[0])
                        
                        # Reconstruct img1 from img2 features
                        pred_img1 = self.model.decoder(feats2)
                        per_sample_error = ((pred_img1 - img1) ** 2).mean(dim=(1, 2, 3))
                    else:
                        # Check if the model has the required predictor
                        if not hasattr(self.model, 'predictor_jepa'):
                            print("Warning: Model doesn't have predictor_jepa, falling back to dummy reconstruction difficulty")
                            return torch.rand(img1.shape[0])
                        
                        # JEPA-style: predict img1 features from img2 features
                        pred_feats1 = self.model.predictor_jepa(feats2)
                        per_sample_error = F.mse_loss(pred_feats1, feats1, reduction='none').mean(dim=1)
                
                # Normalize difficulties to [0, 1] for consistency
                per_sample_error = per_sample_error.cpu()
                if per_sample_error.numel() > 1:
                    min_err = per_sample_error.min()
                    max_err = per_sample_error.max()
                    
                    if max_err > min_err:
                        difficulties = (per_sample_error - min_err) / (max_err - min_err)
                    else:
                        difficulties = torch.zeros_like(per_sample_error)
                else:
                    difficulties = per_sample_error
                
                return difficulties
                
            except Exception as e:
                print(f"Error computing temporal reconstruction difficulty: {e}")
                print("Falling back to random difficulty scores...")
                # Return random difficulties as fallback
                return torch.rand(img1.shape[0])
    
    def _compute_mae_reconstruction_temporal(self, feats_context, feats_target, mask, batch):
        """Compute MAE-style reconstruction with temporal context.
        
        Args:
            feats_context: features from temporal context frame (img2)
            feats_target: features from target frame (img1) that we want to classify
            mask: boolean mask indicating which patches to reconstruct (True = masked)
            batch: unused in this version
        """
        B = feats_context.shape[0]
        
        # Replace masked tokens with mask token
        mask_tokens = self.model.mask_token.repeat(B, mask.sum(dim=1), 1)
        
        # Combine visible context tokens with mask tokens
        x_full = feats_context.clone()
        x_full[mask] = mask_tokens.reshape(-1, feats_context.shape[-1])
        
        # Decode the full sequence to predict target features
        pred_patches = self.model.decoder(x_full)
        
        # Get target patches (actual target features)
        cls_offset = 1 if hasattr(self.model.backbone, "cls_token") else 0
        
        # Use target features for masked positions
        masked_pred = []
        masked_target = []
        
        for i in range(B):
            # Get masked indices, adjusting for CLS token if needed
            mask_indices = mask[i, cls_offset:].nonzero().squeeze(1)
            
            if len(mask_indices.shape) == 0:  # Handle case of single masked patch
                mask_indices = mask_indices.unsqueeze(0)
            
            # Get predicted patches for masked positions
            if mask_indices.numel() > 0:  # Check if there are any masked patches
                masked_pred.append(pred_patches[i, mask_indices])
                # Use actual target features as reconstruction targets
                masked_target.append(feats_target[i, mask_indices + cls_offset])
        
        # Compute reconstruction loss
        if len(masked_pred) > 0:
            masked_pred = torch.cat(masked_pred)
            masked_target = torch.cat(masked_target)
            
            # Compute MSE loss
            loss = F.mse_loss(masked_pred, masked_target)
            
            # Compute per-sample errors
            per_sample_error = torch.zeros(B, device=feats_context.device)
            start_idx = 0
            
            for i in range(B):
                mask_count = mask[i, cls_offset:].sum().item()
                if mask_count > 0:
                    end_idx = start_idx + mask_count
                    sample_pred = masked_pred[start_idx:end_idx]
                    sample_target = masked_target[start_idx:end_idx]
                    per_sample_error[i] = F.mse_loss(sample_pred, sample_target)
                    start_idx = end_idx
        else:
            # Fallback for edge case with no masked tokens
            loss = torch.tensor(0.0, device=feats_context.device)
            per_sample_error = torch.zeros(B, device=feats_context.device)
        
        return loss, per_sample_error
    
    def _compute_jepa_reconstruction_temporal(self, feats_context, feats_target, context_mask, target_mask):
        """Compute JEPA-style prediction with temporal context.
        
        Args:
            feats_context: features from temporal context frame (img2)
            feats_target: features from target frame (img1) that we want to classify
            context_mask: boolean mask indicating which patches are context (True = context)
            target_mask: boolean mask indicating which patches are targets (True = target)
        """
        B = feats_context.shape[0]
        
        # Get context tokens from context frame
        context_tokens = feats_context[context_mask].reshape(B, -1, feats_context.shape[-1])
        
        # Get target tokens from actual target frame
        target_tokens = feats_target[target_mask].reshape(B, -1, feats_target.shape[-1])
        
        # Predict target frame tokens from context frame tokens
        context_pooled = context_tokens.mean(dim=1)
        pred_target = self.model.predictor_jepa(context_pooled).unsqueeze(1).expand_as(target_tokens)
        
        # Compute prediction loss against actual target frame
        loss = F.mse_loss(pred_target, target_tokens)
        
        # Compute per-sample errors
        per_sample_error = torch.mean((pred_target - target_tokens)**2, dim=(1, 2))
        
        return loss, per_sample_error
    
    def compute_entropy_difficulty(self, logits):
        """Compute difficulty based on prediction entropy."""
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            max_entropy = torch.log(torch.tensor(logits.shape[-1], dtype=torch.float32))
            difficulty = entropy / max_entropy
        return difficulty.cpu()
    
    def compute_margin_difficulty(self, logits):
        """Compute difficulty based on prediction margin (difference between top 2 predictions)."""
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            top2_probs, _ = torch.topk(probs, 2, dim=-1)
            margin = top2_probs[:, 0] - top2_probs[:, 1]
            difficulty = 1.0 - margin  # Lower margin = higher difficulty
        return difficulty.cpu()
    
    def evaluate_dataset(self, dataloader, max_batches=None):
        """Evaluate the entire dataset using actual temporal pairs."""
        print(f"Evaluating with {self.difficulty_method} difficulty method using actual temporal pairs...")
        print("Temporal ordering: predicting classification frame (img1) from context frame (img2)")
        
        # Debug: Check the first batch for feature compatibility
        first_batch = True
        
        # Add running accuracy tracker for debugging
        running_correct = 0
        running_total = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Parse batch structure - TemporalCore50 returns ((img1, img2), target)
            (img1, img2), targets = batch
            
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            targets = targets.to(self.device)
            
            # Extract features from img1 for classification (same as during linear training)
            with torch.no_grad():
                out1 = self.model.forward(img1)
                
                # Extract features in the same way as during linear training
                if self.model.is_vit:
                    # For ViT, use the features that would have been used during linear training
                    features = out1.get('feats_vit', out1.get('feats'))
                    
                    if first_batch:
                        print(f"ViT features shape before pooling: {features.shape}")
                    
                    # Handle ViT features - during linear training, features are usually pooled
                    if hasattr(features, 'shape') and len(features.shape) > 2:
                        if features.shape[1] > 1:  # Has sequence dimension (tokens)
                            # Use CLS token if available, otherwise global average pooling
                            if hasattr(self.model.backbone, "cls_token"):
                                features = features[:, 0]  # CLS token
                                if first_batch:
                                    print("Using CLS token for ViT features")
                            else:
                                features = features.mean(dim=1)  # Global average pooling
                                if first_batch:
                                    print("Using global average pooling for ViT features")
                else:
                    # For ResNet/CNN, use the backbone features directly
                    features = out1.get('feats')
                    if features is None:
                        features = out1 if isinstance(out1, torch.Tensor) else out1.get('z', out1)
                    
                    if first_batch:
                        print(f"CNN/ResNet features shape: {features.shape}")
                
                # Ensure features are 2D for the linear classifier
                if len(features.shape) > 2:
                    features = features.view(features.size(0), -1)
                
                if first_batch:
                    print(f"Final features shape for classifier: {features.shape}")
                    print(f"Expected classifier input size: {self.classifier.in_features}")
                    print(f"Classifier output size: {self.classifier.out_features}")
                    
                    # Check feature compatibility
                    if features.shape[1] != self.classifier.in_features:
                        print(f"WARNING: Feature dimension mismatch!")
                        print(f"Model features: {features.shape[1]}, Classifier expects: {self.classifier.in_features}")
                
                # Get predictions from linear classifier
                logits = self.classifier(features)
                predictions = torch.argmax(logits, dim=-1)
                
                if first_batch:
                    print(f"Logits shape: {logits.shape}")
                    print(f"Predictions shape: {predictions.shape}")
                    print(f"Target range: [{targets.min().item()}, {targets.max().item()}]")
                    print(f"Prediction range: [{predictions.min().item()}, {predictions.max().item()}]")
                    
                    # Show some sample predictions vs targets
                    print(f"First 10 predictions: {predictions[:10].cpu().numpy()}")
                    print(f"First 10 targets:     {targets[:10].cpu().numpy()}")
                    
                    first_batch = False
            
            # Compute difficulties based on method
            if self.difficulty_method == 'reconstruction':
                # Predict img1 (classification target) from img2 (temporal context)
                difficulties = self.compute_temporal_reconstruction_difficulty(img1, img2)
            elif self.difficulty_method == 'entropy':
                difficulties = self.compute_entropy_difficulty(logits)
            elif self.difficulty_method == 'margin':
                difficulties = self.compute_margin_difficulty(logits)
            
            # Compute correctness
            correct = (predictions == targets).cpu().numpy()
            
            # Update running accuracy tracker
            batch_correct = correct.sum()
            batch_total = len(correct)
            running_correct += batch_correct
            running_total += batch_total
            
            # Debug output every 10 batches
            if (batch_idx + 1) % 10 == 0:
                running_acc = running_correct / running_total
                batch_acc = batch_correct / batch_total
                print(f"Batch {batch_idx + 1}: Batch acc = {batch_acc:.3f}, Running acc = {running_acc:.3f}")
                
                # Check for sudden accuracy drops
                if running_acc < 0.1 and batch_idx > 10:
                    print(f"WARNING: Accuracy dropped below 10% at batch {batch_idx + 1}!")
                    print(f"Sample predictions: {predictions[:5].cpu().numpy()}")
                    print(f"Sample targets:     {targets[:5].cpu().numpy()}")
                    print(f"Sample difficulties: {difficulties[:5].numpy()}")
            
            # Store results
            self.results['difficulties'].extend(difficulties.numpy())
            self.results['predictions'].extend(predictions.cpu().numpy())
            self.results['targets'].extend(targets.cpu().numpy())
            self.results['correct'].extend(correct)
            self.results['indices'].extend([batch_idx * dataloader.batch_size + i for i in range(len(targets))])
            
            # Store features for first few batches for potential visualization
            if batch_idx < 5:
                if len(features.shape) > 2:
                    features = features.view(features.size(0), -1)
                self.results['features'].extend(features.cpu().numpy())
                
            # Clear GPU cache periodically to prevent memory issues
            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
    
    def analyze_results(self):
        """Analyze the relationship between difficulty and accuracy."""
        # Convert to numpy arrays
        difficulties = np.array(self.results['difficulties'])
        correct = np.array(self.results['correct'])
        predictions = np.array(self.results['predictions'])
        targets = np.array(self.results['targets'])
        
        # Overall statistics
        overall_acc = correct.mean()
        print(f"\nOverall Accuracy: {overall_acc:.4f}")
        print(f"Difficulty Range: [{difficulties.min():.4f}, {difficulties.max():.4f}]")
        print(f"Difficulty Mean: {difficulties.mean():.4f} ± {difficulties.std():.4f}")
        
        # Compute accuracy by difficulty bins
        n_bins = 10
        bin_edges = np.linspace(difficulties.min(), difficulties.max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        bin_accuracies = []
        bin_counts = []
        bin_stds = []
        
        for i in range(n_bins):
            mask = (difficulties >= bin_edges[i]) & (difficulties < bin_edges[i + 1])
            if i == n_bins - 1:  # Include the maximum value in the last bin
                mask = (difficulties >= bin_edges[i]) & (difficulties <= bin_edges[i + 1])
            
            bin_samples = correct[mask]
            if len(bin_samples) > 0:
                bin_accuracies.append(bin_samples.mean())
                bin_counts.append(len(bin_samples))
                bin_stds.append(bin_samples.std() / np.sqrt(len(bin_samples)))  # SEM
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
                bin_stds.append(0)
        
        # Print bin statistics
        print("\nAccuracy by Difficulty Bins:")
        print("Bin\tDifficulty Range\tSamples\tAccuracy")
        for i in range(n_bins):
            print(f"{i+1}\t[{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}]\t"
                  f"{bin_counts[i]}\t{bin_accuracies[i]:.3f}")
        
        # Compute correlation
        try:
            from scipy.stats import spearmanr, pearsonr
            corr_pearson, p_pearson = pearsonr(difficulties, correct)
            corr_spearman, p_spearman = spearmanr(difficulties, correct)
            
            print(f"\nCorrelation Analysis:")
            print(f"Pearson r = {corr_pearson:.4f} (p = {p_pearson:.4e})")
            print(f"Spearman ρ = {corr_spearman:.4f} (p = {p_spearman:.4e})")
        except ImportError:
            print("\nSciPy not available for correlation analysis")
            corr_pearson, p_pearson = 0, 1
            corr_spearman, p_spearman = 0, 1
        
        # Save detailed results
        results_dict = {
            'overall_accuracy': float(overall_acc),
            'difficulty_stats': {
                'min': float(difficulties.min()),
                'max': float(difficulties.max()),
                'mean': float(difficulties.mean()),
                'std': float(difficulties.std())
            },
            'correlations': {
                'pearson': {'r': float(corr_pearson), 'p': float(p_pearson)},
                'spearman': {'rho': float(corr_spearman), 'p': float(p_spearman)}
            },
            'bins': {
                'edges': bin_edges.tolist(),
                'centers': bin_centers.tolist(),
                'accuracies': [float(x) for x in bin_accuracies],
                'counts': [int(x) for x in bin_counts],
                'std_errors': [float(x) for x in bin_stds]
            },
            'method': self.difficulty_method
        }
        
        save_dir = Path('difficulty_analysis')
        save_dir.mkdir(exist_ok=True)
        
        with open(save_dir / f'analysis_results_{self.difficulty_method}.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Create visualizations
        self.create_visualizations(difficulties, correct, bin_centers, 
                                   bin_accuracies, bin_counts, bin_stds)
        
        return results_dict
    
    def create_visualizations(self, difficulties, correct, bin_centers, 
                              bin_accuracies, bin_counts, bin_stds):
        """Create comprehensive visualization plots."""
        save_dir = Path('difficulty_analysis/plots')
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        plt.style.use('default')  # Use default instead of seaborn for compatibility
        
        # 1. Main analysis figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Difficulty distribution by correctness
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.hist(difficulties[correct], bins=50, alpha=0.6, label='Correct', 
                 density=True, color='green')
        ax1.hist(difficulties[~correct], bins=50, alpha=0.6, label='Incorrect', 
                 density=True, color='red')
        ax1.set_xlabel('Difficulty Score')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Distribution of Difficulty Scores by Correctness ({self.difficulty_method.title()})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparison
        ax2 = fig.add_subplot(gs[0, 2])
        box_data = [difficulties[correct], difficulties[~correct]]
        box_labels = ['Correct', 'Incorrect']
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][1].set_facecolor('red')
        for box in bp['boxes']:
            box.set_alpha(0.6)
        ax2.set_ylabel('Difficulty Score')
        ax2.set_title('Difficulty Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Accuracy vs difficulty bins with error bars
        ax3 = fig.add_subplot(gs[1, :])
        bars = ax3.bar(bin_centers, bin_accuracies, 
                       width=0.8*(bin_centers[1]-bin_centers[0]) if len(bin_centers) > 1 else 0.1,
                       yerr=bin_stds, capsize=5, alpha=0.7, edgecolor='black')
        
        # Color bars by accuracy
        norm = plt.Normalize(vmin=0, vmax=1)
        colors = plt.cm.RdYlGn(norm(bin_accuracies))
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)
        
        ax3.set_xlabel('Difficulty Score (Bin Center)')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Classification Accuracy vs Difficulty Bins')
        ax3.set_ylim(0, 1.05)
        ax3.grid(True, alpha=0.3)
        
        # Add sample count annotations
        for i, (x, y, n, err) in enumerate(zip(bin_centers, bin_accuracies, 
                                                bin_counts, bin_stds)):
            if n > 0:
                ax3.text(x, y + err + 0.02, f'n={n}', ha='center', fontsize=8)
        
        # Add overall accuracy line
        overall_acc = correct.mean()
        ax3.axhline(y=overall_acc, color='blue', linestyle='--', alpha=0.7,
                    label=f'Overall Accuracy: {overall_acc:.3f}')
        ax3.legend()
        
        # Scatter plot with trend
        ax4 = fig.add_subplot(gs[2, 0])
        # Subsample for clarity
        n_sample = min(5000, len(difficulties))
        idx = np.random.choice(len(difficulties), n_sample, replace=False)
        
        # Add jitter to y-axis for better visualization
        jitter = np.random.normal(0, 0.02, n_sample)
        ax4.scatter(difficulties[idx], correct[idx] + jitter, alpha=0.3, s=10)
        
        # Add moving average trend line
        sorted_idx = np.argsort(difficulties)
        window_size = len(difficulties) // 20
        moving_avg = []
        x_trend = []
        for i in range(0, len(sorted_idx) - window_size, window_size):
            window = sorted_idx[i:i+window_size]
            x_trend.append(difficulties[window].mean())
            moving_avg.append(correct[window].mean())
        
        ax4.plot(x_trend, moving_avg, 'r-', linewidth=2, label='Moving Average')
        ax4.set_xlabel('Difficulty Score')
        ax4.set_ylabel('P(Correct)')
        ax4.set_title('Difficulty vs Correctness (with trend)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Cumulative accuracy
        ax5 = fig.add_subplot(gs[2, 1])
        sorted_idx = np.argsort(difficulties)
        sorted_correct = correct[sorted_idx]
        cumulative_acc = np.cumsum(sorted_correct) / np.arange(1, len(sorted_correct) + 1)
        percentiles = np.linspace(0, 100, len(cumulative_acc))
        
        ax5.plot(percentiles, cumulative_acc, linewidth=2, color='blue')
        ax5.fill_between(percentiles, cumulative_acc, alpha=0.3, color='blue')
        ax5.set_xlabel('Percentile (Easy → Hard)')
        ax5.set_ylabel('Cumulative Accuracy')
        ax5.set_title('Cumulative Accuracy by Difficulty Percentile')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, 100)
        ax5.set_ylim(0, 1.05)
        
        # Add quartile lines
        for q in [25, 50, 75]:
            idx = int(q * len(cumulative_acc) / 100)
            ax5.axvline(x=q, color='gray', linestyle=':', alpha=0.5)
            ax5.text(q, 0.02, f'{q}%', ha='center', fontsize=8)
            ax5.text(q, cumulative_acc[idx] + 0.02, 
                     f'{cumulative_acc[idx]:.3f}', ha='center', fontsize=8)
        
        # Percentile analysis
        ax6 = fig.add_subplot(gs[2, 2])
        percentile_ranges = ['0-10%', '10-25%', '25-50%', '50-75%', '75-90%', '90-100%']
        percentile_bounds = [0, 10, 25, 50, 75, 90, 100]
        percentile_accs = []
        
        for i in range(len(percentile_bounds) - 1):
            start_idx = int(percentile_bounds[i] * len(sorted_correct) / 100)
            end_idx = int(percentile_bounds[i+1] * len(sorted_correct) / 100)
            if end_idx > start_idx:
                acc = sorted_correct[start_idx:end_idx].mean()
            else:
                acc = 0
            percentile_accs.append(acc)
        
        bars = ax6.bar(percentile_ranges, percentile_accs, alpha=0.7)
        for bar, acc in zip(bars, percentile_accs):
            bar.set_color(plt.cm.RdYlGn(acc))
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax6.set_xlabel('Difficulty Percentile')
        ax6.set_ylabel('Accuracy')
        ax6.set_title('Accuracy by Difficulty Percentiles')
        ax6.set_ylim(0, 1.1)
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Difficulty Analysis - {self.difficulty_method.capitalize()} Method', 
                     fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / f'difficulty_analysis_{self.difficulty_method}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Create a summary infographic
        self.create_summary_infographic(difficulties, correct, overall_acc)
    
    def create_summary_infographic(self, difficulties, correct, overall_acc):
        """Create a clean summary infographic."""
        save_dir = Path('difficulty_analysis/plots')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('white')
        
        # Key metrics
        ax = axes[0, 0]
        ax.axis('off')
        
        # Calculate key statistics
        easy_mask = difficulties <= np.percentile(difficulties, 33)
        medium_mask = (difficulties > np.percentile(difficulties, 33)) & \
                      (difficulties <= np.percentile(difficulties, 67))
        hard_mask = difficulties > np.percentile(difficulties, 67)
        
        easy_acc = correct[easy_mask].mean() if easy_mask.sum() > 0 else 0
        medium_acc = correct[medium_mask].mean() if medium_mask.sum() > 0 else 0
        hard_acc = correct[hard_mask].mean() if hard_mask.sum() > 0 else 0
        
        metrics_text = f"""
Key Metrics ({self.difficulty_method.title()})

Overall Accuracy: {overall_acc:.1%}

Easy Samples (Bottom 33%): {easy_acc:.1%}
Medium Samples (Middle 33%): {medium_acc:.1%}
Hard Samples (Top 33%): {hard_acc:.1%}

Accuracy Drop (Easy→Hard): {(easy_acc - hard_acc):.1%}
Total Samples: {len(difficulties):,}
        """
        
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, 
                fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
        
        # Pie chart of difficulty distribution
        ax = axes[0, 1]
        sizes = [easy_mask.sum(), medium_mask.sum(), hard_mask.sum()]
        labels = ['Easy', 'Medium', 'Hard']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        explode = (0.05, 0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                           autopct='%1.1f%%', explode=explode,
                                           shadow=True, startangle=90)
        ax.set_title('Distribution of Sample Difficulties')
        
        # Accuracy comparison bar chart
        ax = axes[1, 0]
        categories = ['Easy\n(0-33%)', 'Medium\n(33-67%)', 'Hard\n(67-100%)', 'Overall']
        accuracies = [easy_acc, medium_acc, hard_acc, overall_acc]
        colors_bar = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']
        
        bars = ax.bar(categories, accuracies, color=colors_bar, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy by Difficulty Category', fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Difficulty score distribution
        ax = axes[1, 1]
        if correct.sum() > 0 and (~correct).sum() > 0:
            # Create histogram
            bins = np.linspace(difficulties.min(), difficulties.max(), 30)
            ax.hist(difficulties[correct], bins=bins, alpha=0.6, label='Correct', 
                   density=True, color='green')
            ax.hist(difficulties[~correct], bins=bins, alpha=0.6, label='Incorrect', 
                   density=True, color='red')
            ax.set_xlabel('Difficulty Score')
            ax.set_ylabel('Density')
            ax.set_title('Difficulty Distribution by Correctness')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.hist(difficulties, bins=30, alpha=0.7, color='blue')
            ax.set_xlabel('Difficulty Score')
            ax.set_ylabel('Count')
            ax.set_title('Overall Difficulty Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Sample Difficulty Analysis Summary - {self.difficulty_method.title()}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / f'difficulty_summary_{self.difficulty_method}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate sample difficulty and accuracy')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained CurriculumMoCoV3 checkpoint')
    parser.add_argument('--linear_path', type=str, required=True,
                        help='Path to trained linear classifier checkpoint')
    parser.add_argument('--cfg_path', type=str, required=True,
                        help='Path to config file used for training')
    parser.add_argument('--pretrain_cfg_path', type=str, default=None,
                        help='Path to pretraining config file (optional, for better curriculum parameter matching)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--difficulty_method', type=str, default='reconstruction',
                        choices=['reconstruction', 'entropy', 'margin'],
                        help='Method to compute difficulty')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum number of batches to evaluate (for debugging)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    print("=== Difficulty Evaluation Script ===")
    print(f"Model: {args.model_path}")
    print(f"Linear classifier: {args.linear_path}")
    print(f"Config: {args.cfg_path}")
    if args.pretrain_cfg_path:
        print(f"Pretraining config: {args.pretrain_cfg_path}")
    print(f"Difficulty method: {args.difficulty_method}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 40)
    
    # Load the appropriate config for model initialization
    if args.pretrain_cfg_path and Path(args.pretrain_cfg_path).exists():
        print("Using pretraining config for model initialization...")
        model_cfg = OmegaConf.load(args.pretrain_cfg_path)
        data_cfg = OmegaConf.load(args.cfg_path)  # Use linear config for data
        
        # Merge data configuration from linear config into model config
        # This ensures we have the correct num_classes, dataset type, etc.
        if 'data' in data_cfg:
            model_cfg.data = data_cfg.data
            print(f"Using data config from linear config: num_classes={data_cfg.data.get('num_classes', 'unknown')}")
    else:
        print("Using linear config for both model and data...")
        model_cfg = OmegaConf.load(args.cfg_path)
        data_cfg = model_cfg
    
    # Initialize evaluator with model config
    evaluator = DifficultyEvaluator(
        model_path=args.model_path,
        linear_path=args.linear_path,
        cfg_path="dummy",  # We'll set the config directly
        device=args.device,
        difficulty_method=args.difficulty_method
    )
    
    # Override evaluator's cfg with the properly merged config
    evaluator.cfg = model_cfg
    evaluator.data_cfg = data_cfg
    
    # Finalize model loading if it was deferred
    if hasattr(evaluator, 'model_path'):
        evaluator.finalize_model_loading()
    
    # For evaluation, we need to use the temporal dataset to get actual temporal pairs
    # Override dataset to use temporal_core50 regardless of what's in the config
    print("Forcing evaluation to use temporal_core50 dataset for actual temporal pairs...")
    
    # Create a modified config for temporal evaluation
    eval_cfg = OmegaConf.create(data_cfg.data)
    eval_cfg.dataset = "temporal_core50"  # Force temporal dataset
    
    # Add dataset_kwargs if not present
    if 'dataset_kwargs' not in eval_cfg:
        eval_cfg.dataset_kwargs = {}
    
    # Extract time_window from pretrain config if available, otherwise use default
    time_window = 15  # Default
    if args.pretrain_cfg_path and Path(args.pretrain_cfg_path).exists():
        pretrain_cfg = OmegaConf.load(args.pretrain_cfg_path)
        if 'data' in pretrain_cfg and 'dataset_kwargs' in pretrain_cfg.data:
            time_window = pretrain_cfg.data.dataset_kwargs.get('time_window', 15)
    
    eval_cfg.dataset_kwargs.time_window = time_window
    
    # Set backgrounds for evaluation (use val_backgrounds)
    eval_cfg.dataset_kwargs.backgrounds = eval_cfg.get('val_backgrounds', ["s3", "s7", "s10"])
    
    # CRITICAL FIX: Use categories to match linear classifier training
    # Check if linear classifier was trained with categories
    use_categories = False
    if hasattr(data_cfg.data, 'dataset_kwargs') and 'use_categories' in data_cfg.data.dataset_kwargs:
        use_categories = data_cfg.data.dataset_kwargs.use_categories
    
    # Override to use categories if classifier expects 10 classes
    classifier_num_classes = evaluator.classifier.out_features
    if classifier_num_classes == 10:
        use_categories = True
        print(f"Detected 10-class classifier - forcing use_categories=True")
    elif classifier_num_classes == 50:
        use_categories = False
        print(f"Detected 50-class classifier - using instance labels")
    
    eval_cfg.dataset_kwargs.use_categories = use_categories
    
    print(f"Using temporal dataset with time_window={time_window}")
    print(f"Evaluation sessions: {eval_cfg.dataset_kwargs.backgrounds}")
    print(f"Using categories: {use_categories} (classifier expects {classifier_num_classes} classes)")
    
    # Prepare the temporal evaluation dataloader
    from solo.data.custom.temporal_core50 import TemporalCore50
    from torch.utils.data import DataLoader
    from solo.data.pretrain_dataloader import NCropAugmentation
    
    # Create appropriate transforms (similar to pretrain but without strong augmentations)
    
    # Simple transform pipeline for evaluation (minimal augmentation)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create the temporal dataset with correct label type
    eval_dataset = TemporalCore50(
        h5_path=eval_cfg.get('val_path', eval_cfg.get('train_path')),
        transform=lambda img1, img2: (transform(img1), transform(img2)),
        time_window=eval_cfg.dataset_kwargs.time_window,
        backgrounds=eval_cfg.dataset_kwargs.backgrounds,
        use_categories=eval_cfg.dataset_kwargs.use_categories
    )
    
    # Create dataloader
    val_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Temporal evaluation dataset loaded with {len(val_loader)} batches")
    print(f"Dataset contains {len(eval_dataset)} temporal pairs")
    
    # Run evaluation
    evaluator.evaluate_dataset(val_loader, max_batches=args.max_batches)
    
    # Analyze results
    print("\nAnalyzing results...")
    results = evaluator.analyze_results()
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to 'difficulty_analysis/'")
    print(f"Plots saved to 'difficulty_analysis/plots/'")
    print(f"JSON results saved as 'analysis_results_{args.difficulty_method}.json'")


if __name__ == '__main__':
    main() 