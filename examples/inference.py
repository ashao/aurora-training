import torch

from datetime import datetime
from aurora import Aurora, Batch, Metadata

from aurora_training.utils.data import generate_batch

nsamples = 1
ntime = 1
nlat = 64
nlon = 64

batch = generate_batch(nsamples, ntime, nlat, nlon)
model = Aurora().to("cuda")
predict = model.forward(batch)
print(f"Input 2t shape: {batch.surf_vars['2t'].shape}")
print(f"Predicted 2t shape: {predict.surf_vars['2t'].shape}")
