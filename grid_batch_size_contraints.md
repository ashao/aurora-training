# Grid & Batch Size Constraints in High-Resolution Training

## Problem Description
When increasing the spatial resolution from 180×360 to 720×1440 (NLAT & NLON ×4) on a node with 4 A100 GPUs, training the Aurora model under FSDP triggers a CUDA error:

```
RuntimeError: CUDA error: invalid configuration argument
```

This error originates in PyTorch’s `scaled_dot_product_attention` kernel during level aggregation in the Perceiver encoder.

## Root Cause Analysis
- **Token Explosion**: With a 4×4 patch, each sample produces `(720/4)×(1440/4)=64 800` tokens per pressure level.
- **Flattening & Batch Dim**: In `Perceiver3DEncoder.aggregate_levels`, tokens of shape `(B, C_A, L, D)` are flattened to `(B×L, C_A, D)` before attention.
- **Head Count**: Scaled-dot-product launches one CUDA block per (batch×head×query) combination. With 16 heads and B×L ≈ 64 800, blocks ≈ 1 036 800.
- **CUDA Grid Limit**: On A100 (and most GPUs), `gridDim.x` max is 65 535. Launching >65 535 blocks in x dimension triggers “invalid configuration argument.”

---

## Debug & Validation Steps
1. Print `B` and `L` just before flatten in `aggregate_levels` to confirm sizes.
2. Run with `CUDA_LAUNCH_BLOCKING=1` to pinpoint the failing launch.
3. Compile with `TORCH_USE_CUDA_DSA=1` for device-side assertions.

## Mitigation Strategies

1. **Reduce Batch Size**
   - Decrease `--samples-per-batch` to ensure that `B×L×num_heads` remains within 65,535.
   - During my tests, I had to set `--samples-per-batch` to `1` to achieve this.
2. **Chunk Level Aggregation**
   In `aggregate_levels`, split the token dimension `L` into smaller slices:
   ```python
   max_q = 4096  # ensure max_q × num_heads ≤ 65535
   out_chunks = []
   for i in range(0, L, max_q):
       xs = x[:, :, i:i+max_q, :]
       ls = latents[:, :, i:i+max_q, :]
       flat_xs = xs.flatten(0, 1)      # (B2×Q, C_A, D)
       flat_ls = ls.flatten(0, 1)
       flat_out = self.level_agg(flat_ls, flat_xs)
       out = flat_out.unflatten(0, (B2, -1)).permute(0, 2, 1, 3)
       out_chunks.append(out)
   return torch.cat(out_chunks, dim=2)
   ```
3. **Reduce Token Count**
   - Increase patch size (e.g. from 4 to 8) or lower grid resolution.

**Result**: Applying one of these strategies might help keep CUDA grid launches within hardware limits and restore training at full resolution.
