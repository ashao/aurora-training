# Copyright 2025 Andrew Shao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from aurora import Batch, Metadata, normalisation
from datetime import datetime


class MappableBatch(Batch):
    """A mappable version of Batch to be used with Torch Dataloaders"""

    def __len__(self):
        return next(iter(self.surf_vars.values())).shape[0]

    def __getitem__(self, idx):
        return MappableBatch(
            surf_vars={k: self.surf_vars[k][idx, ...] for k in self.surf_vars},
            static_vars=self.static_vars,
            atmos_vars={k: self.atmos_vars[k][idx, ...] for k in self.atmos_vars},
            metadata=self.metadata,
        )

    @classmethod
    def from_batch(cls, batch):
        return MappableBatch(
            surf_vars=batch.surf_vars,
            static_vars=batch.static_vars,
            atmos_vars=batch.atmos_vars,
            metadata=batch.metadata,
        )


class DatasetBatch(torch.utils.data.Dataset):
    """Combine two Batch objects to use with Torch Dataloaders"""

    def __init__(self, inputs: MappableBatch, predictions: MappableBatch):
        self.inputs = inputs
        self.predictions = predictions

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.predictions[idx]


# TODO: Need to add a new normalization routine for Batch. Default
# Aurora Batch has hardcoded values for atmos_vars, but requires
# surf_vars?!


# For now, retrieve the hard-coded stats
def calc_surf_stats(batch):
    return {
        var: (normalisation.locations[var], normalisation.scales[var])
        for var in batch.surf_vars
    }


def generate_batch(nsamples, ntime, nlat, nlon):
    # Generate random data
    return MappableBatch(
        surf_vars={
            k: torch.randn(nsamples, ntime, nlat, nlon)
            for k in ("2t", "10u", "10v", "msl")
        },
        static_vars={k: torch.randn(nlat, nlon) for k in ("lsm", "z", "slt")},
        atmos_vars={
            k: torch.randn(nsamples, ntime, 4, nlat, nlon)
            for k in ("z", "u", "v", "t", "q")
        },
        metadata=Metadata(
            lat=torch.linspace(90, -90, nlat),
            lon=torch.linspace(0, 360, nlon + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )


def collate_mappable_batch(batch):

    inputs = [item[0] for item in batch]
    predictions = [item[1] for item in batch]

    surf_vars = inputs[0].surf_vars
    atmos_vars = inputs[0].atmos_vars

    # Combine the individual MappableBatch objects into a single MappableBatch
    combined_inputs = MappableBatch(
        surf_vars={k: torch.stack([b.surf_vars[k] for b in inputs]) for k in surf_vars},
        static_vars=inputs[0].static_vars,
        atmos_vars={
            k: torch.stack([b.atmos_vars[k] for b in inputs]) for k in atmos_vars
        },
        metadata=inputs[0].metadata,
    )
    combined_predictions = MappableBatch(
        surf_vars={
            k: torch.stack([b.surf_vars[k] for b in predictions]) for k in surf_vars
        },
        static_vars=predictions[0].static_vars,
        atmos_vars={
            k: torch.stack([b.atmos_vars[k] for b in predictions]) for k in atmos_vars
        },
        metadata=predictions[0].metadata,
    )

    return combined_inputs, combined_predictions
