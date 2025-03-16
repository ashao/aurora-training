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

import typing as t

import torch

from aurora import Batch


# Loss function for Aurora as defined in Section D1 Training objective
# in Bodnar et al. [2024]
def loss_function(
    predicted: Batch,
    truth: Batch,
    w_s: t.Dict[str, float],
    w_c: t.Dict[str, float],
    gamma: float = 1.0,
    alpha: float = 0.25,
    beta: float = 1.0,
):
    """Calculate the loss function as defined in the Aurora paper

    :param predicted: The result from a forward pass
    :param truth: The true result
    :param w_s: Weights for each surface variable (w_k^s in the paper)
    :param w_c: Weights for each atmospheric variable (w_{k,c}^A in the paper)
    :param gamma: Weight for the entire dataset, defaults to 1.
    :param alpha: Weight for the surface variable loss, defaults to 0.25
    :param beta: Weight for the atmospheric variable loss, defaults to 1.
    :return: The loss function for this particular dataset
    """

    nlat = predicted.metadata.lat.numel()
    nlon = predicted.metadata.lon.numel()
    nlev = len(predicted.metadata.atmos_levels)

    def sum_abs_error(pred, truth):
        return torch.sum(torch.abs(pred - truth), dim=[0, -1, -2])

    def surface_loss(pred_s, truth_s, w_s):
        device = next(iter(pred_s.values())).device
        loss = torch.tensor(0.0, requires_grad=True, device=device)
        for var in pred_s:
            mae = sum_abs_error(pred_s[var], truth_s[var])
            # Scale by pre-defined weight by variable
            loss = loss + w_s[var] * mae

        # Normalize by the number lat, lon points
        return loss / (nlat * nlon)

    def level_loss(pred_c, truth_c, w_c):
        device = next(iter(pred_c.values())).device
        loss = torch.tensor(0.0, requires_grad=True, device=device)
        for var in pred_c:
            mae = sum_abs_error(pred_c[var], truth_c[var])
            # Scale by pre-defined weight by variable and level
            loss = loss + torch.sum(w_c[var] * mae)

        # Normalize by number of lat, lon, and level points
        return loss / (nlev * nlat * nlon)

    Vs = len(predicted.surf_vars)
    Va = len(predicted.atmos_vars)
    loss_s = surface_loss(predicted.surf_vars, truth.surf_vars, w_s)
    loss_c = level_loss(predicted.atmos_vars, truth.atmos_vars, w_c)
    total = (gamma / (Vs + Va)) * ((alpha * loss_s) + (beta * loss_c))
    return total
