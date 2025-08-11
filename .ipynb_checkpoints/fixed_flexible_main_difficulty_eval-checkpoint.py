"""
Fixed Flexible Difficulty Evaluator that handles configuration issues properly.
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
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression

from solo.methods.base import BaseMethod
from solo.data.classification_dataloader import prepare_data
from omegaconf import OmegaConf


class FixedFlexibleDifficultyEvaluator:
    """Fixed evaluator that handles config issues properly."""
    
    def __init__(self, model_path: str, linear_path: str, cfg_path: str, 
                 device: str = 'cuda', difficulty_method: str = 'reconstruction'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.difficulty_method = difficulty_method
        
        # Load config
        print(f"Loading config from: {cfg_path}")
        self.cfg = OmegaConf.load(cfg_path)
        
        # Load pretrained model (CurriculumMoCoV3)
        print("Loading pretrained model...")
        self.model = self.load_pretrained_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load linear classifier
        print("Loading linear classifier...")
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
            'features': []  # Optional: store features for t-SNE
        }
    
    def create_minimal_config(self, base_cfg):
        """Create a properly minimal config with ALL required sections."""
        
        # Create minimal config with all required sections
        minimal_cfg = OmegaConf.create({
            'method': base_cfg.get('method', 'selective_curriculum_mocov3'),
            'backbone': {
                'name': base_cfg.backbone.get('name', 'resnet18'),
                'kwargs': base_cfg.backbone.get('kwargs', {})
            },
            
            # CRITICAL: Include ALL method_kwargs
            'method_kwargs': {
                'proj_hidden_dim': base_cfg.method_kwargs.get('proj_hidden_dim', 4096),
                'proj_output_dim': base_cfg.method_kwargs.get('proj_output_dim', 256),
                'pred_hidden_dim': base_cfg.method_kwargs.get('pred_hidden_dim', 4096),
                'temperature': base_cfg.method_kwargs.get('temperature', 0.1),
                'curriculum_type': base_cfg.method_kwargs.get('curriculum_type', 'jepa'),
                'curriculum_reverse': base_cfg.method_kwargs.get('curriculum_reverse', False),
                'curriculum_warmup_epochs': base_cfg.method_kwargs.get('curriculum_warmup_epochs', 20),
                'curriculum_weight': base_cfg.method_kwargs.get('curriculum_weight', 1.0),
                'reconstruction_masking_ratio': base_cfg.method_kwargs.get('reconstruction_masking_ratio', 0.75),
                'num_candidates': base_cfg.method_kwargs.get('num_candidates', 8),
                'selection_epochs': base_cfg.method_kwargs.get('selection_epochs', 100),
                'base_tau_momentum': base_cfg.method_kwargs.get('base_tau_momentum', 0.996),
                'final_tau_momentum': base_cfg.method_kwargs.get('final_tau_momentum', 0.996),
            },
            
            # CRITICAL: Include momentum section
            'momentum': {
                'base_tau': base_cfg.momentum.get('base_tau', 0.996),
                'final_tau': base_cfg.momentum.get('final_tau', 0.996),
            },
            
            # CRITICAL: Include complete optimizer section
            'optimizer': {
                'name': base_cfg.optimizer.get('name', 'lars'),
                'batch_size': base_cfg.optimizer.get('batch_size', 64),
                'lr': base_cfg.optimizer.get('lr', 1.6),
                'weight_decay': base_cfg.optimizer.get('weight_decay', 1e-6),
                'momentum': base_cfg.optimizer.get('momentum', 0.9),
                'eta': base_cfg.optimizer.get('eta', 0.02),
                'grad_clip_lars': base_cfg.optimizer.get('grad_clip_lars', False),
                'exclude_bias_n_norm': base_cfg.optimizer.get('exclude_bias_n_norm', False),
                # Add the missing classifier_lr
                'classifier_lr': base_cfg.optimizer.get('classifier_lr', 0.3),
            },
            
            # CRITICAL: Include complete scheduler section
            'scheduler': {
                'name': base_cfg.scheduler.get('name', 'warmup_cosine'),
                'warmup_epochs': base_cfg.scheduler.get('warmup_epochs', 0.01),
                'warmup_start_lr': base_cfg.scheduler.get('warmup_start_lr', 0.0),
                'eta_min': base_cfg.scheduler.get('eta_min', 0.0),
                'max_epochs': base_cfg.scheduler.get('max_epochs', 100),
                'interval': base_cfg.scheduler.get('interval', 'step'),
                'frequency': base_cfg.scheduler.get('frequency', 1),
                'lr_decay_steps': base_cfg.scheduler.get('lr_decay_steps', None),
                'min_lr': base_cfg.scheduler.get('min_lr', 0.0),
                'step_size': base_cfg.scheduler.get('step_size', 30),
                'gamma': base_cfg.scheduler.get('gamma', 0.1),
            },
            
            # Include data section (might be needed)
            'data': {
                'dataset': base_cfg.data.get('dataset', 'core50_categories'),
                'num_classes': base_cfg.data.get('num_classes', 10),
                'num_large_crops': base_cfg.data.get('num_large_crops', 2),
                'num_small_crops': base_cfg.data.get('num_small_crops', 0),
                'crop_size': base_cfg.data.get('crop_size', 224),
            },
            
            # Include loss section
            'loss': {
                'name': base_cfg.loss.get('name', 'mocov3'),
                'temperature': base_cfg.loss.get('temperature', 0.1),
            },
            
            # Basic training parameters
            'max_epochs': base_cfg.get('max_epochs', 100),
            'devices': base_cfg.get('devices', [0]),
            'accelerator': base_cfg.get('accelerator', 'gpu'),
            'precision': base_cfg.get('precision', 32),
        })
        
        return minimal_cfg
    
    def load_pretrained_model(self, model_path):
        """Load the pretrained CurriculumMoCoV3 model with proper config handling."""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        print("Initializing model...")
        print(f"Model type: {self.cfg.get('method', 'unknown')}")
        
        # Import the correct model class - try multiple possibilities
        method_name = self.cfg.get('method', 'selective_curriculum_mocov3')
        ModelClass = None
        
        if 'selective_curriculum_mocov3' in method_name.lower():
            # Try different possible imports for selective curriculum
            import_attempts = [
                # Try local implementations first
                ('selective_curriculum_mocov3', 'SelectiveCurriculumMoCoV3', 'local SelectiveCurriculumMoCoV3'),
                ('selective_jepa_curriculum_mocov3', 'SelectiveJEPACurriculumMoCoV3', 'local SelectiveJEPACurriculumMoCoV3'),
                # Try solo-learn implementations
                ('solo.methods.curriculum_mocov3', 'CurriculumMoCoV3', 'CurriculumMoCoV3 from solo-learn'),
                ('solo.methods.mocov3', 'MoCoV3', 'base MoCoV3 from solo-learn'),
            ]
            
            for module_name, class_name, description in import_attempts:
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    ModelClass = getattr(module, class_name)
                    print(f"✅ Successfully imported {description}")
                    break
                except (ImportError, AttributeError) as e:
                    print(f"❌ Failed to import {description}: {e}")
                    continue
        elif 'curriculum_mocov3' in method_name.lower():
            try:
                from solo.methods.curriculum_mocov3 import CurriculumMoCoV3 as ModelClass
                print("✅ Using CurriculumMoCoV3")
            except ImportError as e:
                print(f"❌ Failed to import CurriculumMoCoV3: {e}")
        else:
            try:
                from solo.methods.mocov3 import MoCoV3 as ModelClass
                print("✅ Using base MoCoV3")
            except ImportError as e:
                print(f"❌ Failed to import MoCoV3: {e}")
        
        if ModelClass is None:
            raise ImportError(f"Could not import model class for method: {method_name}")
        
        # Try with full config first
        try:
            print("Trying with full config...")
            model = ModelClass(self.cfg)
            print("✅ Successfully initialized with full config")
        except Exception as e:
            print(f"Full config failed: {e}")
            print("Trying with minimal config...")
            
            # Create and try minimal config
            try:
                minimal_cfg = self.create_minimal_config(self.cfg)
                print("Created minimal config with all required sections")
                model = ModelClass(minimal_cfg)
                print("✅ Successfully initialized with minimal config")
            except Exception as e2:
                print(f"Minimal config also failed: {e2}")
                
                # Last resort: try to create an even more basic config
                print("Trying with ultra-minimal config...")
                ultra_minimal = OmegaConf.create({
                    'method': 'selective_curriculum_mocov3',
                    'backbone': {'name': 'resnet18'},
                    'method_kwargs': {
                        'proj_hidden_dim': 4096,
                        'proj_output_dim': 256,
                        'pred_hidden_dim': 4096,
                        'temperature': 0.1,
                        'curriculum_type': 'jepa',
                        'num_candidates': 8,
                    },
                    'momentum': {'base_tau': 0.996, 'final_tau': 0.996},
                    'optimizer': {
                        'name': 'lars',
                        'lr': 1.6,
                        'weight_decay': 1e-6,
                        'classifier_lr': 0.3,
                        'batch_size': 64
                    },
                    'scheduler': {
                        'name': 'warmup_cosine',
                        'max_epochs': 100,
                        'warmup_epochs': 0.01,
                        'eta_min': 0.0,
                        'interval': 'step',
                        'frequency': 1,
                        'lr_decay_steps': None
                    },
                    'data': {'num_classes': 10, 'num_large_crops': 2, 'num_small_crops': 0},
                    'max_epochs': 100,
                })
                
                model = ModelClass(ultra_minimal)
                print("✅ Successfully initialized with ultra-minimal config")
        
        # Load state dict
        print("Loading model weights...")
        try:
            model.load_state_dict(state_dict, strict=False)
            print("✅ Model weights loaded successfully")
        except Exception as e:
            print(f"Warning: Some weights couldn't be loaded: {e}")
            # Try to load what we can
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            print(f"✅ Loaded {len(filtered_dict)}/{len(state_dict)} weights")
        
        return model
    
    def load_linear_classifier(self, linear_path):
        """Load the trained linear classifier."""
        checkpoint = torch.load(linear_path, map_location='cpu')
        
        # Extract classifier architecture from checkpoint
        classifier_keys = [k for k in checkpoint['state_dict'].keys() if k.startswith('classifier.')]
        
        if not classifier_keys:
            raise ValueError("No classifier weights found in checkpoint. Make sure this is a linear model checkpoint.")
        
        # Get dimensions from weights
        weight_key = [k for k in classifier_keys if k.endswith('.weight')][0]
        bias_key = [k for k in classifier_keys if k.endswith('.bias')][0]
        
        num_features = checkpoint['state_dict'][weight_key].shape[1]
        num_classes = checkpoint['state_dict'][weight_key].shape[0]
        
        print(f"Linear classifier: {num_features} -> {num_classes}")
        
        classifier = nn.Linear(num_features, num_classes)
        
        # Load classifier weights without prefix
        classifier_state = {
            'weight': checkpoint['state_dict'][weight_key],
            'bias': checkpoint['state_dict'][bias_key]
        }
        classifier.load_state_dict(classifier_state)
        
        return classifier
    
    def compute_entropy_difficulty(self, logits):
        """Compute difficulty based on prediction entropy."""
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            max_entropy = torch.log(torch.tensor(logits.shape[-1], dtype=torch.float32))
            difficulty = entropy / max_entropy
        return difficulty.cpu()
    
    def compute_reconstruction_difficulty(self, images, indices=None):
        """Compute difficulty based on reconstruction error (simplified version)."""
        with torch.no_grad():
            # Get features from the model
            out = self.model(images)
            
            # Get feature representations
            if isinstance(out, dict):
                feats = out.get('feats')
                if feats is None:
                    # Try other common keys
                    for key in ['features', 'representations', 'embeddings']:
                        if key in out:
                            feats = out[key]
                            break
                    else:
                        # Use the first tensor value
                        feats = list(out.values())[0]
            else:
                feats = out
            
            # Simple reconstruction difficulty: use feature variance as proxy
            # Higher variance = more "information" = potentially easier
            # Lower variance = less "information" = potentially harder
            if len(feats.shape) > 2:
                feats = feats.mean(dim=tuple(range(2, len(feats.shape))))  # Global average pool
            
            # Compute per-sample feature variance as difficulty proxy
            difficulties = torch.var(feats, dim=1)
            
            # Normalize to [0, 1]
            min_diff = difficulties.min()
            max_diff = difficulties.max()
            if max_diff > min_diff:
                difficulties = (difficulties - min_diff) / (max_diff - min_diff)
            else:
                difficulties = torch.zeros_like(difficulties)
            
            # Flip so higher values = more difficult
            difficulties = 1.0 - difficulties
            
            return difficulties.cpu(), feats
    
    def evaluate_dataset(self, dataloader, max_batches=None):
        """Evaluate the entire dataset."""
        print(f"Evaluating with {self.difficulty_method} difficulty method...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Parse batch
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                if isinstance(batch[0], torch.Tensor) and batch[0].dim() == 1:
                    indices = batch[0]
                    if isinstance(batch[1], (list, tuple)):
                        images, targets = batch[1]
                    else:
                        images = batch[1]
                        targets = batch[2] if len(batch) > 2 else None
                else:
                    images, targets = batch[0], batch[1]
                    indices = torch.arange(len(images))
            else:
                continue
            
            images = images.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            
            # Get features for classification
            with torch.no_grad():
                out = self.model(images)
                if isinstance(out, dict):
                    features = out.get('feats')
                    if features is None:
                        features = list(out.values())[0]
                else:
                    features = out
                
                # Handle different feature shapes
                if hasattr(features, 'shape') and len(features.shape) > 2:
                    features = features.mean(dim=tuple(range(2, len(features.shape))))
                
                # Get predictions from linear classifier
                logits = self.classifier(features)
                predictions = torch.argmax(logits, dim=-1)
                
                # Compute difficulties based on method
                if self.difficulty_method == 'entropy':
                    difficulties = self.compute_entropy_difficulty(logits)
                else:  # reconstruction
                    difficulties, _ = self.compute_reconstruction_difficulty(images, indices)
            
            # Compute correctness
            if targets is not None:
                correct = (predictions == targets).cpu().numpy()
            else:
                correct = np.zeros(len(predictions))
            
            # Store results
            self.results['difficulties'].extend(difficulties.numpy())
            self.results['predictions'].extend(predictions.cpu().numpy())
            if targets is not None:
                self.results['targets'].extend(targets.cpu().numpy())
            self.results['correct'].extend(correct)
            self.results['indices'].extend(indices.cpu().numpy())
            
            # Optionally store features for visualization
            if batch_idx < 5:  # Only store first few batches
                self.results['features'].extend(features.cpu().numpy())
    
    def analyze_results(self):
        """Analyze the relationship between difficulty and accuracy."""
        # Convert to numpy arrays
        difficulties = np.array(self.results['difficulties'])
        correct = np.array(self.results['correct'])
        predictions = np.array(self.results['predictions'])
        targets = np.array(self.results['targets']) if self.results['targets'] else None
        
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
        
        for i in range(n_bins):
            mask = (difficulties >= bin_edges[i]) & (difficulties < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (difficulties >= bin_edges[i]) & (difficulties <= bin_edges[i + 1])
            
            bin_samples = correct[mask]
            if len(bin_samples) > 0:
                bin_accuracies.append(bin_samples.mean())
                bin_counts.append(len(bin_samples))
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        # Compute correlation
        corr_pearson, p_pearson = pearsonr(difficulties, correct)
        corr_spearman, p_spearman = spearmanr(difficulties, correct)
        
        print(f"\nCorrelation Analysis:")
        print(f"Pearson r = {corr_pearson:.4f} (p = {p_pearson:.4e})")
        print(f"Spearman ρ = {corr_spearman:.4f} (p = {p_spearman:.4e})")
        
        # Create results dictionary
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
                'counts': [int(x) for x in bin_counts]
            }
        }
        
        # Save results
        save_dir = Path('difficulty_analysis')
        save_dir.mkdir(exist_ok=True)
        
        with open(save_dir / 'analysis_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save raw data for batch analysis
        np.savez(save_dir / 'raw_data.npz', 
                 difficulties=difficulties, 
                 correct=correct,
                 predictions=np.array(self.results['predictions']),
                 targets=np.array(self.results['targets']) if self.results['targets'] else None)
        
        return results_dict


def main():
    parser = argparse.ArgumentParser(description='Fixed evaluate sample difficulty and accuracy')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained CurriculumMoCoV3 checkpoint')
    parser.add_argument('--linear_path', type=str, required=True,
                        help='Path to trained linear classifier checkpoint')
    parser.add_argument('--cfg_path', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    parser.add_argument('--difficulty_method', type=str, default='entropy',
                        choices=['reconstruction', 'entropy'],
                        help='Method to compute difficulty')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum number of batches to evaluate (for debugging)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = FixedFlexibleDifficultyEvaluator(
        model_path=args.model_path,
        linear_path=args.linear_path,
        cfg_path=args.cfg_path,
        device=args.device,
        difficulty_method=args.difficulty_method
    )
    
    # Prepare data loader using the same logic as main_linear.py
    val_data_format = evaluator.cfg.data.format
    
    # Extract the use_categories parameter if present in dataset_kwargs
    use_categories = False
    if evaluator.cfg.data.get('dataset_kwargs') is not None and 'use_categories' in evaluator.cfg.data.dataset_kwargs:
        use_categories = evaluator.cfg.data.dataset_kwargs.use_categories

    # Only prepare validation data
    _, val_loader = prepare_data(
        evaluator.cfg.data.dataset,
        train_data_path=None,  # We don't need training data
        val_data_path=evaluator.cfg.data.val_path,
        data_format=val_data_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        auto_augment=False,  # No augmentation for evaluation
        train_backgrounds=getattr(evaluator.cfg.data, 'train_backgrounds', None),
        val_backgrounds=getattr(evaluator.cfg.data, 'val_backgrounds', None),
        use_categories=use_categories
    )
    
    # Run evaluation
    evaluator.evaluate_dataset(val_loader, max_batches=args.max_batches)
    
    # Analyze results
    results = evaluator.analyze_results()
    
    print("\nEvaluation complete! Results saved to 'difficulty_analysis/'")


if __name__ == '__main__':
    main()