# PredifyMoCo: Predictive Coding + Momentum Contrastive Learning

This implementation combines hierarchical predictive coding with momentum contrastive learning, creating a novel self-supervised learning method that leverages both predictive dynamics and contrastive objectives.

## Overview

PredifyMoCo introduces **cross-predictive dynamics** where each layer in the query encoder predicts previous layers from the momentum encoder. This creates a hierarchical predictive system that runs multiple timesteps of dynamics before computing the final contrastive loss.

### Key Features

- **Hierarchical Cross-Prediction**: Each layer predicts the previous layer from the momentum encoder
- **Multi-Timestep Dynamics**: Runs multiple timesteps of predictive updates before final contrastive loss
- **Momentum Targets**: Uses momentum encoder representations as stable prediction targets
- **VGG16 Support**: Specifically designed for VGG16 backbone with appropriate layer hook positions
- **Configurable Hyperparameters**: All predictive coding parameters are configurable

## Architecture

The method consists of:

1. **Query Encoder**: Standard VGG16 backbone for processing current images
2. **Momentum Encoder**: EMA-updated VGG16 for stable target representations
3. **4 PCoder Modules**: Predictive coding modules at different layers
4. **Contrastive Heads**: Standard MoCo-style projector and predictor

### PCoder Architecture

Each PCoder performs the update:
```
rep = β(feedforward) + λ(feedback) + (1-β-λ)(memory) - α(gradient)
```

Where:
- `β` (ffm): Feedforward multiplier
- `λ` (fbm): Feedback multiplier  
- `α` (erm): Error multiplier
- gradient: ∇(prediction_error)

### Layer Mapping (VGG16)

| PCoder | Query Layer | Target Layer | Dimensions | Feedback |
|--------|-------------|--------------|------------|----------|
| PCoder1 | Layer 2 (128) | Layer 1 (64) | 128→64 | ✓ |
| PCoder2 | Layer 3 (256) | Layer 2 (128) | 256→128 | ✓ |
| PCoder3 | Layer 4 (512) | Layer 3 (256) | 512→256 | ✓ |
| PCoder4 | Layer 5 (512) | Layer 4 (512) | 512→512 | ✗ |

*VGG16 layer positions: [3, 8, 15, 22, 29] (ReLU after conv layers)*

## Implementation Files

### Core Implementation
- `solo/utils/pcoder.py` - PCoder, Predictor, and PCoderN classes
- `solo/methods/predify_moco.py` - Main PredifyMoCo method
- `solo/methods/__init__.py` - Method registration

### Configuration
- `scripts/pretrain/cifar/predify_moco.yaml` - Training configuration

### Testing
- `test_predify_moco.py` - Forward pass and architecture test
- `test_predify_training.py` - Training step and gradient test

## Configuration Parameters

```yaml
method_kwargs:
  # Standard MoCo parameters
  proj_hidden_dim: 4096      # Projector hidden dimension
  proj_output_dim: 256       # Projector output dimension
  pred_hidden_dim: 4096      # Predictor hidden dimension
  temperature: 0.2           # Contrastive temperature
  
  # Predictive coding parameters
  timesteps: 4               # Number of predictive dynamics timesteps
  pred_loss_weight: 1.0      # Weight for predictive loss vs contrastive loss
  
  # PCoder hyperparameters (for each of 4 PCoders)
  ffm: [0.8, 0.8, 0.8, 0.8]  # Feedforward multipliers
  fbm: [0.1, 0.1, 0.1, 0.0]  # Feedback multipliers (last has no feedback)
  erm: [0.01, 0.01, 0.01, 0.01] # Error multipliers
```

## Usage

### Training with PredifyMoCo

```bash
# Activate the solo-learn environment
conda activate solo-learn

# Train on CIFAR-10
python main_pretrain.py \
    --config-path scripts/pretrain/cifar \
    --config-name predify_moco.yaml

# Train with custom parameters
python main_pretrain.py \
    --config-path scripts/pretrain/cifar \
    --config-name predify_moco.yaml \
    method_kwargs.timesteps=8 \
    method_kwargs.pred_loss_weight=2.0 \
    optimizer.lr=0.0005
```

### Testing the Implementation

```bash
# Test basic forward pass
python test_predify_moco.py

# Test training step
python test_predify_training.py
```

### Key Results from Tests

**Forward Pass Test:**
- ✅ Model initialization successful
- ✅ Query and momentum outputs correct shapes
- ✅ 5 layer representations collected from both encoders
- ✅ Predictive dynamics running for specified timesteps
- ✅ Prediction errors computed for all PCoders

**Training Test:**
- ✅ Training step completed successfully
- ✅ Combined loss (predictive + contrastive + classification)
- ✅ Gradients computed for all 44 parameters
- ✅ Loss values finite and reasonable

## Method Details

### Training Process

1. **Forward Pass**: Process augmented views through query and momentum encoders
2. **Extract Representations**: Hook intermediate features at 5 VGG16 layers
3. **Predictive Dynamics**: Run timesteps of cross-prediction between encoders
4. **Compute Losses**:
   - Predictive loss: Average prediction errors across timesteps and PCoders
   - Contrastive loss: Standard MoCo InfoNCE between projected features
   - Classification loss: Optional online linear evaluation

### Loss Function

```
Total Loss = α·Predictive_Loss + Contrastive_Loss + Classification_Loss
```

Where:
- `Predictive_Loss = (1/T·N) Σₜ Σᵢ MSE(prediction_i^t, target_i)`
- `Contrastive_Loss = InfoNCE(q₁, k₂) + InfoNCE(q₂, k₁)`
- `α` is the predictive loss weight

### Key Advantages

1. **Hierarchical Learning**: Learns representations at multiple abstraction levels
2. **Stable Targets**: Momentum encoder provides stable prediction targets
3. **Multi-Scale Dynamics**: Captures both local and global predictive relationships
4. **Gradient Flow**: Multiple timesteps provide rich gradient signals
5. **Contrastive Consistency**: Final contrastive loss ensures view invariance

## Extending the Implementation

### Adding New Backbones

To support other architectures:

1. Identify appropriate layer positions for feature extraction
2. Create PCoder modules with correct input/output dimensions
3. Register forward hooks at the identified layers
4. Update the assertion in `__init__` to allow the new backbone

### Hyperparameter Tuning

Key hyperparameters to tune:
- `timesteps`: More timesteps = more dynamics but higher computation
- `pred_loss_weight`: Balance between predictive and contrastive objectives
- `ffm, fbm, erm`: Control the dynamics of each PCoder
- `temperature`: Standard contrastive learning temperature

### Research Directions

1. **Learnable Dynamics**: Make ffm, fbm, erm learnable parameters
2. **Attention-Based Feedback**: Use attention for feedback connections
3. **Adaptive Timesteps**: Dynamically determine number of timesteps
4. **Multi-Scale PCoders**: Different PCoder architectures for different scales

## Dependencies

The implementation relies on:
- PyTorch
- Lightning (PyTorch Lightning)
- solo-learn framework
- torchvision (for VGG16)
- omegaconf (for configuration)

## License

This implementation follows the same MIT license as the solo-learn framework.

## Citation

If you use this implementation, please cite both the solo-learn framework and the original predictive coding papers that inspired this work.

---

**Note**: This implementation is currently optimized for VGG16 on CIFAR-10/100. Extension to other datasets (ImageNet) and backbones (ResNet, ViT) would require adjustments to layer positions and PCoder architectures. 