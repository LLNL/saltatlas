# SOLANET NN-Descent on GPU: what the kernels do, and what was measured

Single-GPU NN-Descent for SOLANET, one source tree targeting NVIDIA (CUDA) and
AMD (HIP). This note covers the two kernels, the optimizations applied to them,
the changes that were tried and rejected, and the limitations that remain.

Measurements are whole-index build time, median of 3 or 5 runs, recall@32
against exact ground truth, on an H100 (Matrix) and an MI300A (Tuolumne).

## The two kernels

| | `find_new_neighbor_candidates` | `update_knng_with_candidates` |
|---|---|---|
| parallelism | one warp per point, teams of 8 per pair | one warp per point |
| job | compute distances, propose candidates | install the good candidates, keep the row sorted |

The main loop runs the neighbour check for `(new, new)` and `(new, old)`, then
one update launch per check.

## Applied to the neighbour check

1. **Team vote.** One `__any_sync` ballot replaces a `shfl_down` chain plus a
   broadcast when asking whether a candidate is already a neighbour.
2. **32-bit induction variable** in the distance loop, in place of `size_t`.
3. **128-bit vector loads** (`float4`) when the dimensionality is a multiple of
   four and the data is aligned.
4. **Register cap** via `__launch_bounds__`, selected by dimensionality: high
   dimensional data spills more and wants a lower blocks-per-SM target. Note
   that the second argument does not mean the same thing on both vendors, so
   the attribute is compiled out on HIP (see
   `SALTATLAS_SOLANET_NND_LAUNCH_CAP`).
5. **Reverse push.** Each computed distance is offered to both endpoints rather
   than only to `nid1`, which makes the third `(old, new)` launch redundant.
   The candidate buffer is widened to absorb the extra arrivals.

Together, 1.49x on sift-128 at k=32 against the unmodified baseline.

## Applied to the update kernel

1. **Hoisted duplicate check.** The merge re-scanned the whole KNNG row for
   every candidate, O((c + k) * k). The candidates are already sorted by ID at
   that point, so the same question is answered by walking the row once and
   binary-searching the candidates, O(k log c). Hits are recorded in a bitmask
   rather than written into the array, because overwriting a slot would break
   the ordering the search depends on.
2. **Merge instead of re-sort.** The merge leaves two sorted runs, an untouched
   ascending prefix and the insertions in descending order. Combining them is
   O(k) where the previous full re-sort was O(k^2).
3. **One warp per point.** The candidate buffer and the KNNG row move into
   shared memory. This was the largest single step, and the win came from
   memory placement rather than from the extra parallelism.
4. **Warp-collective duplicate removal**, and
5. **Warp-collective sorts.** Bitonic sorting networks across the warp, with an
   ID tie-break so that equal distances order deterministically. The serial
   path sorted by ID alone and kept whichever entry landed first.

Together, 1.62x on top of the above: **2.41x overall at k=32 and 2.96x at
k=64** on sift-128.

Points whose candidate count exceeds what the sorting network can take fall
back to the serial path; `update_one_point` exists for that case.

## Measured and rejected

Kept here because the reasoning is reusable, not because the code is.

| change | result | why |
|---|---|---|
| Stage `nid1`'s vector in shared memory | slower | shared and L1 are the same physical unit, so relocating a load does not relieve pressure on it |
| Upper-triangle loop on the symmetric pass | 2.0x slower, then 1.15x after the update kernel got cheaper | halving the distance work forces both endpoints through a scattered reverse push, which costs more than the distances saved |
| Fusing the two neighbour-check passes | 9% slower | plus a candidate-buffer overflow |
| Register cap on the update kernel | neutral | the dependency chain is too long for occupancy to help |
| Skipping the re-sort for untouched rows | neutral | the branch is per thread and the warp is not |

## Cross-vendor

Recall agrees between vendors to within 0.01 pp at matched configuration. The
optimizations are worth more on MI300A than on H100 (3.92x against 2.78x on
sift-128), which narrows the gap between the two from 1.68x to 1.19x.

The ordering reverses with dimensionality. On GIST at 960 dimensions, where the
local join is 98% of runtime and the kernel is bandwidth-bound, MI300A finishes
in 40.8 s against H100's 56.8 s at identical settings.

## Known limitations

**Recall per unit k.** SOLANET needs roughly double the build degree to reach
the recall cuVS GNND reaches, and doubling k costs about 3.2x. This is the
dominant remaining gap on GloVe50 and NYTimes, where per-iteration speed is
already competitive. Raising the sampling rate to 1.0 and tightening the
termination threshold by 800x each move recall by less than 0.35 pp, so this is
a converged-quality problem rather than an effort problem. The likely mechanism
is the fixed-width candidate buffer, which discards on overflow regardless of
candidate quality.

**High dimensionality.** At 960 dimensions the local join is 98% of runtime and
the distance path reads each neighbour's vector from global memory once per
pair. cuVS tiles vectors through shared memory instead. This is the gap on
GIST.

**Half precision.** Inner-product datasets with padded dimensions can put the
signal far below the fp16 noise floor. Distances are computed in fp32 here.

## Reproducing

Each optimization was developed as a separately selectable level so that the
baseline and every intermediate step could be run from one binary. That
scaffolding is not in this code; it is preserved at tag
`experiments/opt-levels` for anyone reproducing the measurements above.

Building for AMD needs CMake newer than the system default, and ROCm on the
prefix path:

```sh
export ROCM_PATH=/opt/rocm-7.2.1
export PATH=$ROCM_PATH/bin:$PATH
export CMAKE_PREFIX_PATH=$ROCM_PATH:$ROCM_PATH/lib/cmake:$CMAKE_PREFIX_PATH
```
