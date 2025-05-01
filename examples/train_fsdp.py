import argparse
import os
import time
import functools
from datetime import datetime

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
    StateDictType,
    BackwardPrefetch,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler

from aurora import Aurora
from aurora.model.swin3d import BasicLayer3D
from aurora_training.training.defaults import w_s, w_c
from aurora_training.training.loss import loss_function
from aurora_training.utils.data import (
    generate_batch,
    calc_surf_stats,
    collate_mappable_batch,
    DatasetBatch,
    MappableBatch,
)


def setup_distributed(backend="nccl"):
    """Set up distributed training environment."""
    # Initialize the distributed environment
    if "SLURM_PROCID" in os.environ:
        # Running on a SLURM cluster
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        node_rank = rank // int(os.environ.get("SLURM_GPUS_ON_NODE", 8))
    else:
        # Fallback for manual setup with torchrun
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        node_rank = rank // torch.cuda.device_count()

    # Set the device for this process
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Initialize the process group if not already initialized
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    
    return rank, world_size, local_rank, device


def create_fsdp_model(model, device, args):
    """Wrap model with FSDP."""
    # Define mixed precision policy
    mp_policy = None
    if args.use_mixed_precision:
        # Check if BFloat16 is supported (preferred for stability)
        bf16_ready = (
            torch.cuda.is_available() and 
            torch.cuda.is_bf16_supported() and
            torch.version.cuda and
            hasattr(torch.cuda, "get_device_capability") and
            torch.cuda.get_device_capability()[0] >= 8  # Ampere GPUs and above
        )
        
        if bf16_ready and args.use_bfloat16:
            # Use bfloat16 for better numerical stability
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                # Keep buffers in fp32 for stable training
                buffer_dtype=torch.float32,
            )
        else:
            # Use safer mixed precision that keeps critical buffers in fp32
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                # Keep buffers in fp32 for stable training
                buffer_dtype=torch.float32,
            )
    
    # Configure CPU offloading
    cpu_offload = None
    if args.cpu_offload:
        cpu_offload = CPUOffload(offload_params=True)
    
    # Configure FSDP auto wrap policy for Aurora's BasicLayer3D blocks
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            BasicLayer3D,
        },
    )
    
    # Set sharding strategy
    sharding_strategy = ShardingStrategy.FULL_SHARD
    if args.sharding_strategy == "shard_grad_op":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif args.sharding_strategy == "no_shard":
        sharding_strategy = ShardingStrategy.NO_SHARD
    
    # Create FSDP model with all configurations at once
    fsdp_model = FSDP(
        model.to(device),
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        device_id=device,
        mixed_precision=mp_policy if mp_policy else None,
        cpu_offload=cpu_offload if cpu_offload else None,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE  # Enable backward prefetch for better performance
    )
    
    # Configure activation checkpointing
    apply_activation_checkpointing(fsdp_model, check_fn=lambda x: isinstance(x, BasicLayer3D))
    
    return fsdp_model


def save_checkpoint(model, optimizer, epoch, loss, args, rank):
    """Save model checkpoint."""
    # Create directory on rank 0 only
    if rank == 0:
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Make sure all processes reach this point before continuing
    torch.distributed.barrier()
    
    # For FSDP, use streaming to CPU to save state dict
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    # All processes need to participate in the state_dict_type context
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        # Only rank 0 actually creates the dictionary and saves
        if rank == 0:
            cpu_state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                'args': args,
            }
            
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(cpu_state, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        else:
            # Non-rank 0 processes still need to call state_dict to participate in 
            # the collective operation, but we discard the result
            _ = model.state_dict()
            _ = optimizer.state_dict()
    
    # Ensure all processes are synchronized after saving
    torch.distributed.barrier()
    
    # Clean up any cached memory
    torch.cuda.empty_cache()


def main(args):
    # Setup distributed environment
    rank, world_size, local_rank, device = setup_distributed()
    
    # Log distributed training info
    if rank == 0:
        print(f"Starting FSDP training with {world_size} processes")
        print(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        print(f"Sharding strategy: {args.sharding_strategy}")
    
    # Create output directory
    if rank == 0 and args.save_model:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Data parameters
    NSAMPLES = args.nsamples
    NEPOCHS = args.nepochs
    NTIME_INPUT = args.ntime_input
    NLAT = args.nlat
    NLON = args.nlon
    SAMPLES_PER_BATCH = int(args.samples_per_batch)
    
    # Generate data on rank 0 and distribute to all ranks
    if rank == 0:
        input = generate_batch(NSAMPLES, NTIME_INPUT, NLAT, NLON)
        truth = generate_batch(NSAMPLES, 1, NLAT, NLON)  # Output always has len(time)=1
        surf_stats = calc_surf_stats(input)
        input = MappableBatch.from_batch(input.unnormalise(surf_stats))
        truth = MappableBatch.from_batch(truth.unnormalise(surf_stats))
        training_dataset = DatasetBatch(input, truth)
        package = [training_dataset, surf_stats]
    else:
        package = [None, None]

    dist.broadcast_object_list(package, src=0)

    if rank != 0:
        training_dataset = package[0]
        surf_stats = package[1]

    # Setup data loading
    sampler = DistributedSampler(training_dataset)
    dataloader = DataLoader(
        training_dataset,
        batch_size=SAMPLES_PER_BATCH,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_mappable_batch,
    )
    
    # Initialize model
    model = Aurora(use_lora=args.use_lora, autocast=True, surf_stats=surf_stats)
    model.train()
    model.configure_activation_checkpointing()
    
    # Wrap model with FSDP
    model = create_fsdp_model(model, device, args)
    
    # Create optimizer
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Training loop
    start_time = time.time()
    best_loss = float('inf')
    best_loss_tensor = torch.tensor([best_loss], device=device)
    
    for epoch in range(NEPOCHS):
        epoch_start = time.time()
        sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{NEPOCHS}")
        
        # Training loop
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_input, batch_truth in dataloader:
            # Move data to device
            batch_input = batch_input.to(device)
            batch_truth = batch_truth.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred = model(batch_input).normalise(surf_stats)
            
            # Calculate loss
            loss = loss_function(
                pred, batch_truth.normalise(surf_stats), w_s, w_c
            )
            
            # Backward pass
            loss.backward()
            
            # Apply gradient clipping
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Update weights
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch progress
            if args.verbose and rank == 0 and num_batches % args.log_interval == 0:
                print(f"  Batch {num_batches}, Loss: {loss.item():.4f}")
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Gather loss from all processes for correct reporting
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size
        
        # Log epoch results
        epoch_time = time.time() - epoch_start
        if rank == 0:
            print(f"  Epoch {epoch+1} completed in {epoch_time:.2f}s | Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint if requested
        if args.save_model and (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, args, rank)
            
        # Save best model - synchronize the best_loss across all processes
        is_best = avg_loss < best_loss
        if args.save_model and is_best:
            # Update best loss
            best_loss = avg_loss
            best_loss_tensor[0] = best_loss
            
            # Broadcast best loss from rank 0 to ensure all processes have the same value
            dist.broadcast(best_loss_tensor, src=0)
            best_loss = best_loss_tensor.item()
            
            # Save best checkpoint
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, args, rank)
    
    # Log final training stats
    if rank == 0:
        total_time = time.time() - start_time
        print(f"{NEPOCHS} epochs completed in {total_time:.2f}s")
    
    # Save final checkpoint
    if args.save_model:
        save_checkpoint(model, optimizer, NEPOCHS, avg_loss, args, rank)
    
    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Aurora model with FSDP")
    # Data parameters
    parser.add_argument(
        "--nsamples",
        help="The total number of samples to generate for training",
        default=12,
        type=int,
    )
    parser.add_argument(
        "--nepochs", 
        help="The number of epochs to train for", 
        default=5, 
        type=int,
    )
    parser.add_argument(
        "--ntime_input", 
        help="The number of time levels in the input", 
        default=2,
        type=int,
    )
    parser.add_argument(
        "--nlat", 
        help="Number of zonal points", 
        default=180,
        type=int,
    )
    parser.add_argument(
        "--nlon", 
        help="Number of meridional points", 
        default=360,
        type=int,
    )
    
    # Training parameters
    parser.add_argument(
        "--optimizer", 
        help="The name of the optimizer to use", 
        default="Adam",
    )
    parser.add_argument(
        "--lr",
        help="Learning rate",
        default=3e-4,
        type=float,
    )
    parser.add_argument(
        "--weight_decay",
        help="Weight decay",
        default=0.01,
        type=float,
    )
    parser.add_argument(
        "--samples-per-batch", 
        help="Number of samples to draw per batch", 
        default=1,
        type=int,
    )
    parser.add_argument(
        "--use_lora", 
        help="Use LoRA for parameter-efficient training", 
        action="store_true",
    )
    parser.add_argument(
        "--max_grad_norm",
        help="Maximum gradient norm for clipping",
        default=1.0,
        type=float,
    )
    
    # FSDP parameters
    parser.add_argument(
        "--sharding_strategy",
        help="FSDP sharding strategy",
        choices=["full_shard", "shard_grad_op", "no_shard"],
        default="full_shard",
    )
    parser.add_argument(
        "--use_mixed_precision",
        help="Use mixed precision for training",
        action="store_true",
    )
    parser.add_argument(
        "--use_bfloat16",
        help="Use bfloat16 precision (requires Ampere or newer GPUs)",
        action="store_true",
    )
    parser.add_argument(
        "--cpu_offload",
        help="Offload parameters to CPU when not in use",
        action="store_true",
    )
    
    # Logging and saving parameters
    parser.add_argument(
        "--verbose", 
        help="Print verbose training information", 
        action="store_true",
    )
    parser.add_argument(
        "--save_model", 
        help="Save model checkpoint after training", 
        action="store_true",
    )
    parser.add_argument(
        "--output_dir", 
        help="Directory to save checkpoints", 
        default="./output",
        type=str,
    )
    parser.add_argument(
        "--log_interval", 
        help="Logging interval in batches", 
        default=10,
        type=int,
    )
    parser.add_argument(
        "--save_interval", 
        help="Checkpoint saving interval in epochs", 
        default=1,
        type=int,
    )
    
    args = parser.parse_args()
    main(args)