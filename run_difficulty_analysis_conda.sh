#!/bin/bash

# Comprehensive Difficulty Analysis Script for CurriculumMoCoV3
# This script runs all three difficulty analysis methods and generates comprehensive results

echo "=== CurriculumMoCoV3 Difficulty Analysis ==="
echo "Starting analysis at $(date)"

# Activate conda environment
echo "Activating solo-learn conda environment..."
conda activate solo-learn

# Configuration paths
MODEL_PATH="/home/brothen/solo-learn/trained_models/curriculum_mocov3/mae-exponential-bs3x64/mocov3-curriculum-mae-exponential-core50-bs3x64-np6j8ah8-ep=20-stp=13125.ckpt"
LINEAR_PATH="/home/brothen/solo-learn/trained_models/linear/linear/2bocrcbv/mae_exponential_bs64x3_ep20_categories-2bocrcbv-ep=9-stp=18750.ckpt"
CFG_PATH="/home/brothen/solo-learn/scripts/linear_probe/core50/mocov3.yaml"
PRETRAIN_CFG_PATH="/home/brothen/solo-learn/scripts/pretrain/core50/mocov3_curriculum_exponential.yaml"

# Analysis parameters
BATCH_SIZE=64
NUM_WORKERS=4
DEVICE="cuda"
# MAX_BATCHES=50  # Removed - process full validation set

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Linear: $LINEAR_PATH" 
echo "  Config: $CFG_PATH"
echo "  Pretraining Config: $PRETRAIN_CFG_PATH"
echo "  Batch Size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo ""

# Check if files exist
echo "Verifying files exist..."
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "❌ Error: Model checkpoint not found at $MODEL_PATH"
    exit 1
fi

if [[ ! -f "$LINEAR_PATH" ]]; then
    echo "❌ Error: Linear checkpoint not found at $LINEAR_PATH"
    exit 1
fi

if [[ ! -f "$CFG_PATH" ]]; then
    echo "❌ Error: Config file not found at $CFG_PATH"
    exit 1
fi

if [[ ! -f "$PRETRAIN_CFG_PATH" ]]; then
    echo "❌ Error: Pretraining config file not found at $PRETRAIN_CFG_PATH"
    exit 1
fi

echo "✅ All files found!"
echo ""

# Method 1: Reconstruction-based difficulty
echo "🔄 Running reconstruction-based difficulty analysis..."
python evaluate_difficulty.py \
    --model_path "$MODEL_PATH" \
    --linear_path "$LINEAR_PATH" \
    --cfg_path "$CFG_PATH" \
    --pretrain_cfg_path "$PRETRAIN_CFG_PATH" \
    --difficulty_method reconstruction \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --device $DEVICE

if [[ $? -eq 0 ]]; then
    echo "✅ Reconstruction analysis completed successfully"
else
    echo "❌ Reconstruction analysis failed"
fi
echo ""

# Method 2: Entropy-based difficulty
echo "🔄 Running entropy-based difficulty analysis..."
python evaluate_difficulty.py \
    --model_path "$MODEL_PATH" \
    --linear_path "$LINEAR_PATH" \
    --cfg_path "$CFG_PATH" \
    --pretrain_cfg_path "$PRETRAIN_CFG_PATH" \
    --difficulty_method entropy \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --device $DEVICE

if [[ $? -eq 0 ]]; then
    echo "✅ Entropy analysis completed successfully"
else
    echo "❌ Entropy analysis failed"
fi
echo ""

# Method 3: Margin-based difficulty
echo "🔄 Running margin-based difficulty analysis..."
python evaluate_difficulty.py \
    --model_path "$MODEL_PATH" \
    --linear_path "$LINEAR_PATH" \
    --cfg_path "$CFG_PATH" \
    --pretrain_cfg_path "$PRETRAIN_CFG_PATH" \
    --difficulty_method margin \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --device $DEVICE

if [[ $? -eq 0 ]]; then
    echo "✅ Margin analysis completed successfully"
else
    echo "❌ Margin analysis failed"
fi
echo ""

# Summary
echo "=== Analysis Complete ==="
echo "Results saved to:"
echo "  📊 Plots: difficulty_analysis/plots/"
echo "  📄 Data: difficulty_analysis/analysis_results_*.json"
echo ""

echo "Generated files:"
ls -la difficulty_analysis/plots/
ls -la difficulty_analysis/analysis_results_*.json

echo ""
echo "Analysis completed at $(date)"
echo "=== END ===" 