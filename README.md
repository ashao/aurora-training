# Overview

This repository includes scripts, extra classes, and utility functions to train
Microsoft's Aurora model. Examples are also included to demonstrate inference and
distributed training va DDP.

This is still under development and subject to frequent changes. Cloning from Github to
pull changes rapidly and then installing in editable is thus highly recommended:
```
git clone https://github.com/ashao/aurora-training.git
pip install -e aurora-training/
```

To run the inference example:
```
cd examples
python inference.py
```

To run the training example using DDP (assuming 1 node and 4 GPUs):
```
cd examples/
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
         train_ddp.py --nsamples 12 --samples-per-batch 4
```

To run the training using FSDP on one or more nodes, please refer to [document](README_fsdp.md)
