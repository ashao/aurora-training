# Multi-Node FSDP Training for Aurora

This guide explains how to run Fully Sharded Data Parallel (FSDP) training for the Aurora model across multiple nodes, each with multiple GPUs.

## Prerequisites

- Multiple compute nodes with CUDA-capable GPUs
- Network connectivity between all nodes 
- Same number of GPUs on each node
- NCCL support for inter-node communication
- PyTorch with distributed training support

## Run Single-Node Training

For single-node training with multiple GPUs, use the `run_fsdp_singlenode.sh` script:

```bash
./run_fsdp_singlenode.sh --num_gpus 4 --save_model
```

### Basic Usage

```bash
cd examples
./run_fsdp_singlenode.sh [OPTIONS]
```

### Common Options

- `--num_gpus`: Number of GPUs to use (default: 4)
- `--nsamples`: Total number of samples to generate (default: 256)
- `--samples-per-batch`: Samples per gpu per batch (default: 4)
- `--nepochs`: Number of training epochs (default: 3)
- `--save_model`: Enable model checkpoint saving
- `--cpu_offload`: Offload parameters to CPU when not in use

### Example

```bash
./run_fsdp_singlenode.sh \
  --num_gpus 4 \
  --nsamples 128 \
  --samples-per-batch 8 \
  --nepochs 3 \
  --save_model \
  --output_dir "./output/my_experiment"
```


## Running Multi-Node Training

The multi-node training script (`run_fsdp_multinode.sh`) must be executed on each participating node with appropriate parameters.

### Basic Usage

You need to run the script on each node with these mandatory parameters:

```bash
# On node 0 (master node)
./run_fsdp_multinode.sh --nnodes 2 --node_rank 0 --master_addr <MASTER_IP> --master_port 29500 --save_model

# On node 1
./run_fsdp_multinode.sh --nnodes 2 --node_rank 1 --master_addr <MASTER_IP> --master_port 29500 --save_model
```

### Required Parameters

- `--nnodes`: Total number of nodes participating in training
- `--node_rank`: Rank of the current node (0 to nnodes-1)
- `--master_addr`: IP address of the master node (node with rank 0)  
- `--master_port`: Available port on the master node for coordination

### Common Training Configuration

These parameters should be consistent across all nodes:

```bash
--num_gpus_per_node 4 \
--nsamples 24 \
--samples-per-batch 3 \
--nepochs 10 \
--nlat 180 \
--nlon 360 \
--sharding_strategy full_shard \
--output_dir ./output \
```

## Example: 2-Node Training with 4 GPUs per Node

Start training on the master node (192.168.1.100):

```bash
./run_fsdp_multinode.sh \
  --nnodes 2 \
  --node_rank 0 \
  --master_addr 192.168.1.100 \
  --master_port 29500 \
  --num_gpus_per_node 4 \
  --nsamples 24 \
  --samples-per-batch 3 \
  --nepochs 10 \
  --use_mixed_precision \
  --use_bfloat16 \
  --save_model
```

Then, on the second node:

```bash
./run_fsdp_multinode.sh \
  --nnodes 2 \
  --node_rank 1 \
  --master_addr 192.168.1.100 \
  --master_port 29500 \
  --num_gpus_per_node 4 \
  --nsamples 24 \
  --samples-per-batch 3 \
  --nepochs 10 \
  --use_mixed_precision \
  --use_bfloat16 \
  --save_model
```

## Using SLURM in HPC Environments

If you're using SLURM, you can create a submission script like this:

```bash
#!/bin/bash
#SBATCH --job-name=aurora_fsdp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00

# Get the node hostnames
NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODES_ARRAY=($NODES)

# Get the first node as the master node
MASTER_NODE=${NODES_ARRAY[0]}
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w $MASTER_NODE hostname --ip-address)
MASTER_PORT=29500

# Run the distributed training
srun ./run_fsdp_multinode.sh \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --num_gpus_per_node=4 \
  --nsamples 24 \
  --samples-per-batch 6 \
  --nepochs 10 \
  --use_mixed_precision \
  --use_bfloat16 \
  --save_model
```

## Troubleshooting

1. **NCCL errors**: Set `export NCCL_DEBUG=INFO` to get more diagnostic information
2. **Network issues**: Ensure all nodes can communicate with each other
3. **Different configurations**: Make sure all nodes use the same training parameters
4. **PyTorch version mismatch**: Use the same PyTorch version across all nodes
5. **GPU memory OOM**: Try reducing the batch size or use CPU offloading

For more help, check the logs in the `logs/` directory on each node.

### Known Issue - Mixed Precision Options NOT Fully Functioning Yet

Aurora model requires special handling for mixed precision:
- `--use_mixed_precision`: Enables mixed precision training
- `--use_bfloat16`: BFloat16 is recommended for Aurora as it keeps critical buffers in FP32

There is a longitude range issue with mixed precision when running the command `./run_fsdp_singlenode.sh --use_mixed_precision --use_bfloat16`:
```
[rank0]: Traceback (most recent call last):
[rank0]:   File "/lustre/xuco/workspace/llm/aurora-training/examples/train_fsdp.py", line 453, in <module>
[rank0]:     main(args)
[rank0]:   File "/lustre/xuco/workspace/llm/aurora-training/examples/train_fsdp.py", line 260, in main
[rank0]:     pred = model(batch_input).normalise(surf_stats)
[rank0]:            ^^^^^^^^^^^^^^^^^^
[rank0]:   File "/lustre/xuco/software/miniconda3/envs/aurora/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/lustre/xuco/software/miniconda3/envs/aurora/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/lustre/xuco/software/miniconda3/envs/aurora/lib/python3.11/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py", line 848, in forward
[rank0]:     args, kwargs = _root_pre_forward(self, self, args, kwargs)
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/lustre/xuco/software/miniconda3/envs/aurora/lib/python3.11/site-packages/torch/distributed/fsdp/_runtime_utils.py", line 596, in _root_pre_forward
[rank0]:     return _root_cast_forward_input(state, module, args, kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/lustre/xuco/software/miniconda3/envs/aurora/lib/python3.11/site-packages/torch/distributed/fsdp/_runtime_utils.py", line 614, in _root_cast_forward_input
[rank0]:     args, kwargs = _cast_forward_inputs(input_dtype, *args, **kwargs)
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/lustre/xuco/software/miniconda3/envs/aurora/lib/python3.11/site-packages/torch/distributed/utils.py", line 74, in _cast_forward_inputs
[rank0]:     return (_apply_to_tensors(cast_fn, args), _apply_to_tensors(cast_fn, kwargs))
[rank0]:             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/lustre/xuco/software/miniconda3/envs/aurora/lib/python3.11/site-packages/torch/distributed/utils.py", line 263, in _apply_to_tensors
[rank0]:     return apply(container)
[rank0]:            ^^^^^^^^^^^^^^^^
[rank0]:   File "/lustre/xuco/software/miniconda3/envs/aurora/lib/python3.11/site-packages/torch/distributed/utils.py", line 259, in apply
[rank0]:     return type(x)(apply(el) for el in x)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/lustre/xuco/software/miniconda3/envs/aurora/lib/python3.11/site-packages/torch/distributed/utils.py", line 259, in <genexpr>
[rank0]:     return type(x)(apply(el) for el in x)
[rank0]:                    ^^^^^^^^^
[rank0]:   File "/lustre/xuco/software/miniconda3/envs/aurora/lib/python3.11/site-packages/torch/distributed/utils.py", line 241, in apply
[rank0]:     changes = {
[rank0]:               ^
[rank0]:   File "/lustre/xuco/software/miniconda3/envs/aurora/lib/python3.11/site-packages/torch/distributed/utils.py", line 242, in <dictcomp>
[rank0]:     f.name: apply(getattr(dc, f.name)) for f in dataclasses.fields(dc)
[rank0]:             ^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/lustre/xuco/software/miniconda3/envs/aurora/lib/python3.11/site-packages/torch/distributed/utils.py", line 244, in apply
[rank0]:     return dataclasses.replace(dc, **changes)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/lustre/xuco/software/miniconda3/envs/aurora/lib/python3.11/dataclasses.py", line 1503, in replace
[rank0]:     return obj.__class__(**changes)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "<string>", line 8, in __init__
[rank0]:   File "/lustre/xuco/software/miniconda3/envs/aurora/lib/python3.11/site-packages/aurora/batch.py", line 49, in __post_init__
[rank0]:     raise ValueError("Longitudes must be in the range [0, 360).")
[rank0]: ValueError: Longitudes must be in the range [0, 360).
```
