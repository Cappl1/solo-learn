# CurriculumMoCoV3 Difficulty Analysis Toolkit

This toolkit evaluates the relationship between sample difficulty and classification accuracy for CurriculumMoCoV3 models trained with curriculum learning.

## Overview

The toolkit provides three different methods to measure sample difficulty:

1. **Reconstruction** - Uses the curriculum model's reconstruction error (MAE/JEPA)
2. **Entropy** - Uses prediction uncertainty from the linear classifier
3. **Margin** - Uses confidence gap between top 2 predictions

## Quick Start

### Prerequisites

1. Trained CurriculumMoCoV3 model checkpoint
2. Trained linear classifier checkpoint  
3. Data configuration files
4. `solo-learn` conda environment

### Basic Usage

```bash
# Activate environment
conda activate solo-learn

# Run reconstruction difficulty analysis
python evaluate_difficulty.py \
    --model_path "/path/to/curriculum_model.ckpt" \
    --linear_path "/path/to/linear_classifier.ckpt" \
    --cfg_path "/path/to/linear_config.yaml" \
    --pretrain_cfg_path "/path/to/pretrain_config.yaml" \
    --difficulty_method reconstruction
```

### Complete Analysis (All Methods)

```bash
# Run all three difficulty methods
./run_difficulty_analysis_conda.sh
```

## Configuration

### Required Files

1. **Model Checkpoint** - CurriculumMoCoV3 pretrained model with reconstruction components
2. **Linear Checkpoint** - Trained linear classifier on top of frozen features
3. **Linear Config** - Configuration used for linear evaluation (contains data paths and num_classes)
4. **Pretraining Config** - Configuration used for pretraining (contains curriculum parameters)

### Example Configuration

```yaml
# Linear config (difficulty_cur.yaml)
data:
  dataset: "core50_categories"
  train_path: "/path/to/core50_arr.h5"
  val_path: "/path/to/core50_arr.h5"
  format: "h5"
  num_classes: 10  # For category-level evaluation
  val_backgrounds: ["s3", "s7", "s10"]
  dataset_kwargs:
    use_categories: True
```

## Output

### Directory Structure
```
difficulty_analysis/
├── analysis_results_reconstruction.json  # Detailed numerical results
├── analysis_results_entropy.json
├── analysis_results_margin.json
└── plots/
    ├── difficulty_analysis_*.png         # Comprehensive analysis plots
    └── difficulty_summary_*.png          # Summary infographics
```

### Key Metrics

- **Overall Accuracy** - Classification accuracy on validation set
- **Difficulty Statistics** - Min, max, mean, std of difficulty scores
- **Correlations** - Pearson and Spearman correlations between difficulty and correctness
- **Bin Analysis** - Accuracy breakdown by difficulty percentiles

## Results Example

From our Core50 category analysis:

### Reconstruction Method
- **Overall Accuracy**: 75.0%
- **Difficulty Range**: [0.000, 1.000]
- **Correlation**: r = -0.408 (p < 0.001)

**Key Finding**: Strong negative correlation shows reconstruction error effectively predicts classification difficulty.

### Entropy Method  
- **Correlation**: r = -0.263 (p < 0.01)
- More uniform difficulty distribution

### Margin Method
- **Correlation**: r = -0.271 (p < 0.01)
- Wider difficulty range with high variance

## Interpretation

### Reconstruction Difficulty
- **High values** = Hard to reconstruct = Difficult samples
- **Low values** = Easy to reconstruct = Easy samples
- Best correlation with actual classification difficulty

### Entropy Difficulty
- **High values** = Uncertain predictions = Difficult samples
- **Low values** = Confident predictions = Easy samples
- Reflects classifier's confidence directly

### Margin Difficulty  
- **High values** = Small margin between top predictions = Difficult samples
- **Low values** = Large margin = Easy samples
- Most sensitive to classifier decision boundaries

## Advanced Usage

### Custom Difficulty Methods

You can add new difficulty computation methods by extending the `DifficultyEvaluator` class:

```python
def compute_custom_difficulty(self, images, logits):
    """Implement your custom difficulty measure."""
    # Your difficulty computation logic
    return difficulties
```

### Batch Processing

For large datasets, use batching:

```bash
python evaluate_difficulty.py \
    --max_batches 100 \  # Limit for testing
    --batch_size 32 \    # Smaller batches for memory
    # ... other args
```

### Different Datasets

The toolkit works with any dataset supported by solo-learn. Key requirements:

1. Update `data.dataset` in config
2. Ensure correct `num_classes` 
3. Provide appropriate data paths
4. Set dataset-specific parameters in `dataset_kwargs`

## Troubleshooting

### Common Issues

1. **Config Key Missing**: The script adds fallback parameters automatically
2. **Model Loading Errors**: Ensure model and linear classifier are compatible
3. **CUDA Memory**: Reduce batch size if running out of GPU memory
4. **Path Issues**: Verify all file paths exist and are accessible

### Error Messages

- `Missing key num_classes`: Linear config needs `data.num_classes`
- `Shape mismatch`: Model and classifier have incompatible dimensions
- `FileNotFoundError`: Check data file paths in config

### Debug Mode

Run with small batch limit to test:

```bash
python evaluate_difficulty.py \
    --max_batches 2 \
    # ... other args
```

## Technical Details

### Model Requirements

- **CurriculumMoCoV3** with reconstruction components (decoder/predictor_jepa)
- **Compatible backbone** (ResNet18/ViT supported)
- **Curriculum type** automatically detected from checkpoint

### Implementation Notes

- Reconstruction uses same masking strategy as training
- Features are extracted with gradients disabled
- Difficulty scores normalized to [0,1] range
- Supports both instance-level and category-level evaluation

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@misc{curriculum_difficulty_analysis,
  title={CurriculumMoCoV3 Difficulty Analysis Toolkit},
  author={Your Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 