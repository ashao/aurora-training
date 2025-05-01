#!/bin/bash
# Script to run the fixed FSDP training on A100 GPUs

# Set environment variables
# export NCCL_DEBUG=INFO

# Default parameters
NUM_GPUS=4
NSAMPLES=256
SAMPLES_PER_BATCH=4
NEPOCHS=3
NLAT=180
NLON=360
NTIME_INPUT=2
OPTIMIZER="Adam"
SHARDING_STRATEGY="full_shard"
LR=3e-4
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
OUTPUT_DIR="./output"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --num_gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --nsamples)
      NSAMPLES="$2"
      shift 2
      ;;
    --samples-per-batch)
      SAMPLES_PER_BATCH="$2"
      shift 2
      ;;
    --nepochs)
      NEPOCHS="$2"
      shift 2
      ;;
    --nlat)
      NLAT="$2"
      shift 2
      ;;
    --nlon)
      NLON="$2"
      shift 2
      ;;
    --ntime_input)
      NTIME_INPUT="$2"
      shift 2
      ;;
    --optimizer)
      OPTIMIZER="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --weight_decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    --max_grad_norm)
      MAX_GRAD_NORM="$2"
      shift 2
      ;;
    --sharding_strategy)
      SHARDING_STRATEGY="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --use_lora)
      USE_LORA="--use_lora"
      shift
      ;;
    --use_mixed_precision)
      USE_MIXED_PRECISION="--use_mixed_precision"
      shift
      ;;
    --use_bfloat16)
      USE_BFLOAT16="--use_bfloat16"
      shift
      ;;
    --cpu_offload)
      CPU_OFFLOAD="--cpu_offload"
      shift
      ;;
    --verbose)
      VERBOSE="--verbose"
      shift
      ;;
    --save_model)
      SAVE_MODEL="--save_model"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create log directory
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/fsdp_run_${TIMESTAMP}.log"

# Print configuration
echo "=== FSDP Training Configuration ==="
echo "Number of GPUs: $NUM_GPUS"
echo "Samples: $NSAMPLES"
echo "Samples per batch: $SAMPLES_PER_BATCH"
echo "Epochs: $NEPOCHS"
echo "Grid: ${NLAT}x${NLON}"
echo "Time inputs: $NTIME_INPUT"
echo "Optimizer: $OPTIMIZER (lr=${LR}, weight_decay=${WEIGHT_DECAY})"
echo "Max gradient norm: $MAX_GRAD_NORM"
echo "Sharding strategy: $SHARDING_STRATEGY"
echo "Output directory: $OUTPUT_DIR"
echo "Mixed precision: ${USE_MIXED_PRECISION:+enabled}${USE_MIXED_PRECISION:+${USE_BFLOAT16:+ (bfloat16)}${USE_BFLOAT16:-' (float16)'}}"
echo "CPU offload: ${CPU_OFFLOAD:+enabled}"
echo "Use LoRA: ${USE_LORA:+enabled}"
echo "Verbose logging: ${VERBOSE:+enabled}"
echo "Save model: ${SAVE_MODEL:+enabled}"
echo "Log file: $LOG_FILE"
echo "====================================="

# Run the training script with torchrun
echo "Starting FSDP training with $NUM_GPUS GPUs, logs will be saved to $LOG_FILE"

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
  train_fsdp.py \
  --nsamples $NSAMPLES \
  --samples-per-batch $SAMPLES_PER_BATCH \
  --nepochs $NEPOCHS \
  --nlat $NLAT \
  --nlon $NLON \
  --ntime_input $NTIME_INPUT \
  --optimizer $OPTIMIZER \
  --lr $LR \
  --weight_decay $WEIGHT_DECAY \
  --max_grad_norm $MAX_GRAD_NORM \
  --sharding_strategy $SHARDING_STRATEGY \
  --output_dir $OUTPUT_DIR \
  $USE_MIXED_PRECISION \
  $USE_BFLOAT16 \
  $CPU_OFFLOAD \
  $USE_LORA \
  $VERBOSE \
  $SAVE_MODEL \
  2>&1 | tee $LOG_FILE

echo "Training completed. Log saved to $LOG_FILE"