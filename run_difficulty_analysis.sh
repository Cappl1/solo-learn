#!/bin/bash

# Difficulty Analysis Runner Script
# This script runs the difficulty evaluation with your specific setup

# Set paths (adjust these to match your actual paths)
MODEL_PATH="/home/brothen/solo-learn/trained_models/curriculum_mocov3/mae-exponential-bs3x64/mocov3-curriculum-mae-exponential-core50-bs3x64-np6j8ah8-ep=20-stp=13125.ckpt"
LINEAR_PATH="/home/brothen/solo-learn/trained_models/linear/linear/2bocrcbv/mae_exponential_bs64x3_ep20_categories-2bocrcbv-ep=9-stp=18750.ckpt"  # Update this with your actual linear model path
CONFIG_PATH="/home/brothen/solo-learn/scripts/linear_probe/core50/difficulty_cur.yaml"

# Analysis parameters
BATCH_SIZE=64
NUM_WORKERS=4
DEVICE="cuda"

echo "=== Running Difficulty Analysis ==="
echo "This will analyze the relationship between sample difficulty and classification accuracy"
echo ""

# Check if files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model checkpoint not found at $MODEL_PATH"
    echo "Please update MODEL_PATH in this script"
    exit 1
fi

if [ ! -f "$LINEAR_PATH" ]; then
    echo "ERROR: Linear classifier checkpoint not found at $LINEAR_PATH"
    echo "Please update LINEAR_PATH in this script with the path to your trained linear classifier"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found at $CONFIG_PATH"
    echo "Please update CONFIG_PATH in this script"
    exit 1
fi

echo "Found all required files:"
echo "  Model: $MODEL_PATH"
echo "  Linear: $LINEAR_PATH"
echo "  Config: $CONFIG_PATH"
echo ""

# Create output directory
mkdir -p difficulty_analysis

# Run reconstruction-based difficulty analysis
echo "Running reconstruction-based difficulty analysis..."
python evaluate_difficulty.py \
    --model_path "$MODEL_PATH" \
    --linear_path "$LINEAR_PATH" \
    --cfg_path "$CONFIG_PATH" \
    --difficulty_method reconstruction \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --device $DEVICE

echo ""
echo "Reconstruction analysis complete!"
echo ""

# Run entropy-based difficulty analysis
echo "Running entropy-based difficulty analysis..."
python evaluate_difficulty.py \
    --model_path "$MODEL_PATH" \
    --linear_path "$LINEAR_PATH" \
    --cfg_path "$CONFIG_PATH" \
    --difficulty_method entropy \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --device $DEVICE

echo ""
echo "Entropy analysis complete!"
echo ""

# Run margin-based difficulty analysis
echo "Running margin-based difficulty analysis..."
python evaluate_difficulty.py \
    --model_path "$MODEL_PATH" \
    --linear_path "$LINEAR_PATH" \
    --cfg_path "$CONFIG_PATH" \
    --difficulty_method margin \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --device $DEVICE

echo ""
echo "Pixel entropy analysis complete!"
echo ""

# Run pixel entropy-based difficulty analysis
echo "Running pixel entropy-based difficulty analysis..."
python evaluate_difficulty.py \
    --model_path "$MODEL_PATH" \
    --linear_path "$LINEAR_PATH" \
    --cfg_path "$CONFIG_PATH" \
    --difficulty_method pixel_entropy \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --device $DEVICE

echo ""
echo "=== All analyses complete! ==="
echo ""
echo "Results saved to:"
echo "  JSON files: difficulty_analysis/analysis_results_*.json"
echo "  Plots: difficulty_analysis/plots/"
echo ""
echo "Generated plots:"
echo "  - difficulty_analysis_reconstruction.png: Comprehensive reconstruction analysis"
echo "  - difficulty_analysis_entropy.png: Comprehensive entropy analysis"
echo "  - difficulty_analysis_margin.png: Comprehensive margin analysis"
echo "  - difficulty_analysis_pixel_entropy.png: Comprehensive pixel entropy analysis"
echo "  - difficulty_summary_*.png: Summary infographics for each method"
echo ""
echo "Open the PNG files to view the analysis results!" 