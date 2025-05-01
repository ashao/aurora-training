#!/bin/bash
# Script to run FSDP training on multiple nodes, each with the same number of GPUs

# Usage info
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Run FSDP training across multiple nodes"
  echo
  echo "Mandatory options:"
  echo "  --nnodes NUM              Number of nodes for training (required)"
  echo "  --node_rank RANK          Rank of this node (0 to nnodes-1) (required)"
  echo "  --master_addr HOST        IP address of the master node (required)"
  echo "  --master_port PORT        Port on the master node for coordination (required)"
  echo
  echo "Optional arguments:"
  echo "  --num_gpus_per_node NUM   Number of GPUs per node (default: all available)"
  echo "  --nsamples NUM            Number of samples for training (default: 12)"
  echo "  --samples-per-batch NUM   Samples per batch (default: 3)"
  echo "  --nepochs NUM             Number of training epochs (default: 10)"
  echo "  --nlat NUM                Number of latitude points (default: 180)"
  echo "  --nlon NUM                Number of longitude points (default: 360)"
  echo "  --ntime_input NUM         Number of time inputs (default: 2)"
  echo "  --optimizer NAME          Optimizer to use (default: Adam)"
  echo "  --lr NUM                  Learning rate (default: 3e-4)"
  echo "  --weight_decay NUM        Weight decay factor (default: 0.01)"
  echo "  --max_grad_norm NUM       Max gradient norm for clipping (default: 1.0)"
  echo "  --sharding_strategy STR   FSDP sharding strategy (default: full_shard)"
  echo "  --output_dir PATH         Directory to save outputs (default: ./output)"
  echo "  --use_lora                Enable LoRA for training"
  echo "  --use_mixed_precision     Enable mixed precision training"
  echo "  --cpu_offload             Enable CPU offloading"
  echo "  --verbose                 Enable verbose logging"
  echo "  --save_model              Enable model saving"
  echo "  --help                    Display this help message and exit"
  exit 1
}

# Check for help flag
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage
fi

# Required parameters check
if [[ -z "$1" ]]; then
  echo "Error: Required parameters missing"
  usage
fi

# Set environment variables for distributed training
export NCCL_DEBUG=INFO

# Default parameters
NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NSAMPLES=12
SAMPLES_PER_BATCH=3
NEPOCHS=10
NLAT=180
NLON=360
NTIME_INPUT=2
OPTIMIZER="Adam"
SHARDING_STRATEGY="full_shard"
LR=3e-4
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
OUTPUT_DIR="./output"

# Required parameters (must be set by user)
NNODES=""
NODE_RANK=""
MASTER_ADDR=""
MASTER_PORT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --nnodes)
      NNODES="$2"
      shift 2
      ;;
    --node_rank)
      NODE_RANK="$2"
      shift 2
      ;;
    --master_addr)
      MASTER_ADDR="$2"
      shift 2
      ;;
    --master_port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --num_gpus_per_node)
      NUM_GPUS_PER_NODE="$2"
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
      usage
      ;;
  esac
done

# Check if required parameters are provided
if [[ -z "$NNODES" || -z "$NODE_RANK" || -z "$MASTER_ADDR" || -z "$MASTER_PORT" ]]; then
  echo "Error: Required parameters missing"
  echo "NNODES: $NNODES"
  echo "NODE_RANK: $NODE_RANK"
  echo "MASTER_ADDR: $MASTER_ADDR"
  echo "MASTER_PORT: $MASTER_PORT"
  usage
fi

# Create log directory
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/fsdp_node${NODE_RANK}_${TIMESTAMP}.log"

# Set distributed environment variables
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Print configuration
echo "=== Multi-Node FSDP Training Configuration ==="
echo "Number of nodes: $NNODES"
echo "This node rank: $NODE_RANK"
echo "Master node address: $MASTER_ADDR"
echo "Master node port: $MASTER_PORT"
echo "Number of GPUs per node: $NUM_GPUS_PER_NODE"
echo "Total GPUs: $((NNODES * NUM_GPUS_PER_NODE))"
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
echo "=============================================="

# Run the training script with torchrun
echo "Starting multi-node FSDP training with $NUM_GPUS_PER_NODE GPUs on this node (rank $NODE_RANK), logs will be saved to $LOG_FILE"

torchrun \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
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

echo "Node $NODE_RANK: Training completed. Log saved to $LOG_FILE"