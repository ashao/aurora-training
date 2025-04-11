import argparse
import os
import time

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from aurora import Aurora
from aurora_training.training.defaults import w_s, w_c
from aurora_training.training.loss import loss_function
from aurora_training.utils.data import (
    generate_batch,
    calc_surf_stats,
    collate_mappable_batch,
    DatasetBatch,
    MappableBatch,
)

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def main(args):
    NSAMPLES = args.nsamples
    NEPOCHS = args.nepochs
    NTIME_INPUT = args.ntime_input
    NLAT = args.nlat
    NLON = args.nlon
    OPTIMIZER = getattr(torch.optim, args.optimizer)
    SAMPLES_PER_BATCH = int(args.samples_per_batch)

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    if rank == 0:
	 # force NSAMPLES to be an int to deal with type errors
	 NSAMPLES = int(NSAMPLES)
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

    sampler = DistributedSampler(training_dataset)
    dataloader = DataLoader(
        training_dataset,
        batch_size=SAMPLES_PER_BATCH,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_mappable_batch,
    )

    model = Aurora(use_lora=False, autocast=True, surf_stats=surf_stats)

    model = model.to(rank)
    model.train()
    model.configure_activation_checkpointing()
    model = DDP(model, device_ids=[rank])

    optimizer = OPTIMIZER(model.parameters())

    start = time.time()
    for epoch in range(NEPOCHS):
        sampler.set_epoch(epoch)
        if rank == 0:
            print(f"Epoch {epoch}/{NEPOCHS}")
        for batch_input, batch_truth in dataloader:
            optimizer.zero_grad()
            pred = model.forward(batch_input).normalise(surf_stats)
            loss = loss_function(
                pred, batch_truth.normalise(surf_stats).to(rank), w_s, w_c
            )
            loss.backward()
            optimizer.step()

    if rank == 0:
        print(f"{NEPOCHS} epochs completed in {time.time() - start}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nsamples",
        help="The total number of samples to generate for training",
        default=12,
    )
    parser.add_argument(
        "--nepochs", help="The number of epochs to train for", default=5, required=False
    )
    parser.add_argument(
        "--ntime_input", help="The number of time levels in the input", default=2
    )
    parser.add_argument("--nlat", help="Number of zonal points", default=180)
    parser.add_argument("--nlon", help="Number of meridional points", default=360)
    parser.add_argument(
        "--optimizer", help="The name of the optimizer to use", default="Adam"
    )
    parser.add_argument(
        "--samples-per-batch", help="Number of samples to draw per batch", default=1
    )
    args = parser.parse_args()
    main(args)
