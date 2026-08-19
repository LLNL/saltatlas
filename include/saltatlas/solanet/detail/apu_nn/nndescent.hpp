// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <unistd.h>
#include <cstdio>
#ifndef SALTATLAS_SOLANET_APU_NND_PREFETCH_SRC_FV
#define SALTATLAS_SOLANET_APU_NND_PREFETCH_SRC_FV
#endif

#ifndef SALTATLAS_SOLANET_APU_NND_TEAM_SIZE
#define SALTATLAS_SOLANET_APU_NND_TEAM_SIZE 4
#endif

// Minimum blocks per SM requested from the compiler at kOptLevel >= 5.
// Registers otherwise limit find_new_neighbor_candidates to 9 blocks (about 57
// registers per thread), and its dominant stall is waiting on L1TEX, which more
// resident warps would hide. Capping registers trades spills for occupancy, so
// the useful value is empirical: raise it until local-memory spilling appears.
// The useful cap falls as dimensionality rises, because the vectorised distance
// loop keeps proportionally more float4s live. Measured on H100, whole-index
// build time:
//
//                  uncapped   12 blocks   10 blocks   9 blocks
//   sift    (128d)   3.05 s      2.96 s      3.14 s     3.07 s
//   nytimes (256d)   1.68 s      1.94 s      1.65 s     1.52 s
//
// so the cap is selected at launch from the feature dimensionality. Two data
// points on one GPU; the fallback at any size is the uncapped level 4 path.
// From level 6 the neighbour check pushes each distance to both endpoints'
// candidate lists within a single pass, where the old scheme spread them over
// two passes with a buffer reset in between. The per-point buffer therefore has
// to hold roughly twice as many entries, and overflow is discarded silently by
// whoever loses the atomic race, which costs recall rather than time.
// Expressed as a fraction of k, because the useful value sits between 1 (recall
// collapses to 97.34%) and 2 (the extra single-threaded sorting in
// update_knng_with_candidates eats most of the speedup).
#ifndef SALTATLAS_SOLANET_APU_NND_CANDIDATE_WIDTH_NUM
#define SALTATLAS_SOLANET_APU_NND_CANDIDATE_WIDTH_NUM 2
#endif
#ifndef SALTATLAS_SOLANET_APU_NND_CANDIDATE_WIDTH_DEN
#define SALTATLAS_SOLANET_APU_NND_CANDIDATE_WIDTH_DEN 1
#endif

// Blocks per SM requested for update_knng_with_candidates at level >= 8. That
// kernel is latency-bound: on the launches that dominate, compute sits near 4%
// and memory near 52% with nothing saturated, because each thread runs a serial
// O(k^2) selection sort. More resident warps is the only lever that does not
// require restructuring it.
#ifndef SALTATLAS_SOLANET_APU_NND_UPDATE_MIN_BLOCKS_PER_SM
#define SALTATLAS_SOLANET_APU_NND_UPDATE_MIN_BLOCKS_PER_SM 16
#endif

#ifndef SALTATLAS_SOLANET_APU_NND_MIN_BLOCKS_LOW_DIM
#define SALTATLAS_SOLANET_APU_NND_MIN_BLOCKS_LOW_DIM 12
#endif
#ifndef SALTATLAS_SOLANET_APU_NND_MIN_BLOCKS_HIGH_DIM
#define SALTATLAS_SOLANET_APU_NND_MIN_BLOCKS_HIGH_DIM 9
#endif
#ifndef SALTATLAS_SOLANET_APU_NND_MIN_BLOCKS_DIM_THRESHOLD
#define SALTATLAS_SOLANET_APU_NND_MIN_BLOCKS_DIM_THRESHOLD 128
#endif

// The register cap is a CUDA-only optimization for now.
//
// __launch_bounds__ takes the same two arguments on both backends but they do
// not mean the same thing. On CUDA the second is minBlocksPerMultiprocessor; on
// HIP it is MIN_WARPS_PER_EXECUTION_UNIT, whose useful range on CDNA is roughly
// 1 to 8. The values above were tuned as blocks per SM on an H100, so on AMD
// they would either be rejected or silently request something unrelated to what
// was measured.
//
// Rather than guess at an AMD equivalent, the capped entry points carry no
// launch bound there, which makes them identical to the ordinary ones, so the
// launcher needs no backend-specific branch. Levels 5 to 8 therefore mean "the
// same body without the register cap" on AMD. Every other optimization in the
// ladder stays identical across the two vendors, and the cap becomes a separate
// question to be measured on CDNA in its own right.
#if defined(__CUDACC__)
#define SALTATLAS_SOLANET_NND_LAUNCH_CAP(min_blocks) \
  __launch_bounds__(k_nnd_block_size, min_blocks)
#else
#define SALTATLAS_SOLANET_NND_LAUNCH_CAP(min_blocks)
#endif

#ifndef SALTATLAS_SOLANET_APU_NND_TOP_CANDIDATES
#define SALTATLAS_SOLANET_APU_NND_TOP_CANDIDATES 1
#endif

// #ifndef SALTATLAS_SOLANET_APU_NND_PROFILE
// #define SALTATLAS_SOLANET_APU_NND_PROFILE
// #endif

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#if !defined(__CUDACC__)
#include <hip/hip_runtime.h>
#endif
#include <spdlog/spdlog.h>

#include "saltatlas/solanet/detail/apu_nn/algorithm.hpp"
#include "saltatlas/solanet/detail/apu_nn/distance.hpp"
#include "saltatlas/solanet/detail/apu_nn/hash_table.hpp"
#include "saltatlas/solanet/detail/apu_nn/matrix.hpp"
#include "saltatlas/solanet/detail/apu_nn/memory.hpp"
#include "saltatlas/solanet/detail/apu_nn/utils.hpp"
#include "saltatlas/solanet/singleton_time_recorder.hpp"

namespace saltatlas::solanet::apu_nn {

namespace {
static constexpr int k_warp_size      = k_native_warp_size;
static constexpr int k_nnd_block_size = 128;
static constexpr int k_team_size      = SALTATLAS_SOLANET_APU_NND_TEAM_SIZE;
static constexpr int k_top_candidates =
    SALTATLAS_SOLANET_APU_NND_TOP_CANDIDATES;
#if SALTATLAS_SOLANET_APU_NND_TOP_CANDIDATES != 0 && \
    SALTATLAS_SOLANET_APU_NND_TOP_CANDIDATES != 1 && \
    SALTATLAS_SOLANET_APU_NND_TOP_CANDIDATES != 2 && \
    SALTATLAS_SOLANET_APU_NND_TOP_CANDIDATES != 4 && \
    SALTATLAS_SOLANET_APU_NND_TOP_CANDIDATES != 8
#warning \
    "SALTATLAS_SOLANET_APU_NND_TOP_CANDIDATES is recommended to be 0, 1, 2, 4, or 8."
#endif

struct l2_distance_op {
  template <typename FEType>
  SALTATLAS_HD_DEVICE inline acc_type_t<FEType> operator()(
      const FEType* a, const FEType* b, const size_t dims) const {
    return l2(a, b, dims);
  }

  // kNarrowIdx forwards to l2_team; see the comment there. false is the
  // original behaviour.
  template <typename FEType, int TEAM_SIZE, bool kNarrowIdx = false,
            bool kVecLoad = false>
  SALTATLAS_HD_DEVICE inline acc_type_t<FEType> team(const FEType* a,
                                                     const FEType* b,
                                                     const size_t  dims) const {
    return l2_team<FEType, TEAM_SIZE, kNarrowIdx, kVecLoad>(a, b, dims);
  }
};

struct cosine_distance_op {
  template <typename FEType>
  SALTATLAS_HD_DEVICE inline acc_type_t<FEType> operator()(
      const FEType* a, const FEType* b, const size_t dims) const {
    return alt_cosine(a, b, dims);
  }

  // kNarrowIdx is accepted for interface parity with l2_distance_op but is not
  // yet honoured here; alt_cosine_team still uses a size_t induction variable.
  template <typename FEType, int TEAM_SIZE, bool kNarrowIdx = false,
            bool kVecLoad = false>
  SALTATLAS_HD_DEVICE inline acc_type_t<FEType> team(const FEType* a,
                                                     const FEType* b,
                                                     const size_t  dims) const {
    return alt_cosine_team<FEType, TEAM_SIZE, kNarrowIdx, kVecLoad>(a, b, dims);
  }
};

struct inner_product_distance_op {
  template <typename FEType>
  SALTATLAS_HD_DEVICE inline acc_type_t<FEType> operator()(
      const FEType* a, const FEType* b, const size_t dims) const {
    return inner_product(a, b, dims);
  }

  // kNarrowIdx is accepted for interface parity with l2_distance_op but is not
  // yet honoured here; inner_product_team still uses a size_t induction
  // variable.
  template <typename FEType, int TEAM_SIZE, bool kNarrowIdx = false,
            bool kVecLoad = false>
  SALTATLAS_HD_DEVICE inline acc_type_t<FEType> team(const FEType* a,
                                                     const FEType* b,
                                                     const size_t  dims) const {
    return inner_product_team<FEType, TEAM_SIZE, kNarrowIdx, kVecLoad>(a, b,
                                                                      dims);
  }
};

struct nnd_profile_data {
  size_t n_updates{0};
#ifdef SALTATLAS_SOLANET_APU_NND_PROFILE
  size_t n_distance_cals{0};
#endif
};

// Profile data for neighbor checking step
struct nbr_ck_profile_data {
#ifdef SALTATLAS_SOLANET_APU_NND_PROFILE
  size_t n_unique_candidates{0};
  size_t rm_dup_time{0};
  size_t calc_dist_time{0};

  size_t update_knng_time{0};
  size_t update_knng_rm_dup_time{0};
  size_t update_knng_merge_time{0};

  size_t n_batches{0};
#endif
};

// For candidates bloom filter
// Number of bits in bloom filter for candidate checking
// TODO: Make this configurable or calculate based on k and batch size
// constexpr int k_n_bf_hashes = 4;
// constexpr int k_n_bf_bits   = 8192;
// static_assert(k_n_bf_bits % 8 == 0, "Bloom filter bits must be multiple of
// 8");

// Note: kernel parameters are passed by value. Reference parameters would
// make the device dereference a host stack address, which only works on
// unified-memory APUs and crashes on discrete GPUs.
template <typename IDType, typename FEType, typename DistType, typename DistOp>
__global__ void init_knng(const matrix_view<FEType> pstore, const int k,
                          const uint64_t seed, matrix_view<IDType> knng_ids,
                          matrix_view<DistType> knng_dists) {
  const size_t sid      = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n_points = pstore.n_rows();
  const size_t dims     = pstore.n_cols();
  if (sid >= n_points) {
    return;
  }

  const DistOp  dist_op{};
  const FEType* svec = pstore(sid);
  IDType*       ids  = knng_ids(sid);
  DistType*     dsts = knng_dists(sid);

  rnd_state_type state;
  rnd_init(sid, seed, state);

  for (size_t i = 0; i < k; ++i) {
    IDType nid = static_cast<IDType>(rnd_next(state) % n_points);
    bool   ok  = false;
    while (!ok) {
      ok = (nid != static_cast<IDType>(sid));
      for (size_t j = 0; ok && j < i; ++j) {
        ok = (clear_msb(ids[j]) != nid);
      }
      if (!ok) {
        nid = static_cast<IDType>(rnd_next(state) % n_points);
      }
    }
    ids[i]             = set_msb(nid);
    const FEType* nvec = pstore(nid);
    const auto    d    = dist_op(svec, nvec, dims);
    dsts[i]            = static_cast<DistType>(d);
  }

  // Sort neighbors by distance
  // Small distance first
  sort_neighbors_single_thread(ids, dsts, k, true);

#ifndef NDEBUG
  // Check no duplicate neighbors
  for (size_t i = 0; i < k - 1; ++i) {
    for (size_t j = i + 1; j < k; ++j) {
      assert(clear_msb(ids[i]) != clear_msb(ids[j]));
    }
  }
#endif
}

// Prepare new and old neighbor lists
template <typename IDType>
__global__ void set_new_old(matrix_view<IDType> knng_ids, const int p_new,
                            const int p_old, matrix_view<IDType> new_ng,
                            span<int> new_counts, matrix_view<IDType> old_ng,
                            span<int> old_counts) {
  const size_t sid      = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n_points = knng_ids.n_rows();
  const size_t k        = knng_ids.n_cols();
  if (sid >= n_points) {
    return;
  }

  // Pick up up to p new and old neighbors
  assert(new_ng.n_cols() >= static_cast<size_t>(p_new));
  assert(old_ng.n_cols() >= static_cast<size_t>(p_old));

  IDType* knn_ids_row = knng_ids(sid);
  IDType* news        = new_ng(sid);
  IDType* olds        = old_ng(sid);

  // Initialize full rows with invalid values to guard reverse-neighbor scans.
  static constexpr auto k_invalid_id = std::numeric_limits<IDType>::max();
  const int             new_cap      = static_cast<int>(new_ng.n_cols());
  const int             old_cap      = static_cast<int>(old_ng.n_cols());
  for (int i = 0; i < new_cap; ++i) {
    news[i] = k_invalid_id;
  }
  for (int i = 0; i < old_cap; ++i) {
    olds[i] = k_invalid_id;
  }

  int new_count = 0;
  int old_count = 0;
  for (size_t i = 0; i < k; ++i) {
    const IDType nid = clear_msb(knn_ids_row[i]);
    if (get_msb(knn_ids_row[i])) {
      if (new_count < p_new) {
        news[new_count] = nid;
        // Clear the msb flag as it has been selected
        knn_ids_row[i] = nid;
        ++new_count;
      }
    } else {
      if (old_count < p_old) {
        olds[old_count] = nid;
        ++old_count;
      }
    }
    if (new_count >= p_new && old_count >= p_old) {
      break;
    }
  }
  new_counts[sid] = new_count;
  old_counts[sid] = old_count;
}

// Reverse new and old neighbor lists, adding elements to original lists.
// However, original lists are not expanded here to avoid reallocation.
// After this kernel, 'counts_wk' will hold values that are larger than actual
// number of neighbors.
template <typename IDType>
__global__ void add_reverse_neighbors(matrix_view<IDType> ng_ids,
                                      span<int>           n_neighbors,
                                      span<int>           counts_wk) {
  const size_t sid      = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n_points = ng_ids.n_rows();
  if (sid >= n_points) {
    return;
  }
  static constexpr auto k_invalid_id = std::numeric_limits<IDType>::max();
  const auto            max_capacity = static_cast<int>(ng_ids.n_cols());
  IDType*               nbs          = ng_ids(sid);
  const int             n            = n_neighbors[sid];

  for (int i = 0; i < n; ++i) {
    const IDType nid = nbs[i];
    if (nid == k_invalid_id) {
      break;
    }

    // This block is just to reduce #of writes to counts_wk
    if (counts_wk[nid] >= max_capacity) {
      // nid's neighbor list is full
      continue;
    }

    // Atomically increment nid's count
    const int pos = atomicAdd(&counts_wk[nid], 1);
    if (pos >= max_capacity) {
      // nid's neighbor list is full
      continue;
    }
    IDType* ng_list = ng_ids(nid);
    ng_list[pos]    = sid;
  }
}

// Level 13: one warp per point, list staged in shared memory.
//
// The kernel above is the same shape update_knng_with_candidates had before
// level 12: one thread per point running an O(n^2) selection sort with random
// access straight into global memory. It is 14.7% of the build at level 12,
// second only to the two local-join passes and the update kernel, and it is
// pure bookkeeping.
//
// The fix that worked there works here. Staging the list into shared memory
// moves the sort's working set to a latency roughly twenty times lower, and the
// load and store become coalesced because 32 lanes read consecutive elements of
// one row rather than 32 threads reading 32 different rows. The sort itself is
// still lane 0's serial loop; parallelising it is a later step, and level 12
// showed placement matters far more than the division of work here.
template <typename IDType, bool kBitonicSort = false>
__global__ void remove_duplicate_neighbors_warp_kernel(
    matrix_view<IDType> ng_ids) {
  // Elements per lane for the bitonic network at level 14. warp_bitonic_sort
  // supports 1..8, so the largest power-of-two run it can sort is
  // 8 * warp width: 256 on NVIDIA, 512 on AMD. Longer lists fall back to the
  // serial sort.
  constexpr int k_bitonic_m = 8;
  extern __shared__ char smem_dedup[];
  static constexpr auto  k_invalid_id = std::numeric_limits<IDType>::max();

  const int lane            = static_cast<int>(threadIdx.x) % warpSize;
  const int warp_in_block   = static_cast<int>(threadIdx.x) / warpSize;
  const int width           = static_cast<int>(ng_ids.n_cols());
  int       s_capacity      = 1;
  while (s_capacity < width) {
    s_capacity <<= 1;
  }
  IDType* const s_ng =
      reinterpret_cast<IDType*>(smem_dedup) + warp_in_block * s_capacity;

  const size_t g_warp =
      (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warpSize;
  const size_t n_warps =
      (static_cast<size_t>(gridDim.x) * blockDim.x) / warpSize;
  if (n_warps == 0) {
    return;
  }

  for (size_t sid = g_warp; sid < ng_ids.n_rows(); sid += n_warps) {
    IDType* const g_ng = ng_ids(sid);

    // Find the first invalid ID cooperatively, and stage only the valid prefix.
    // These lists are allocated for the worst case after reverse edges, so the
    // width is usually far larger than the number of valid entries; copying the
    // whole row in and out costs more than the sort it saves. Each lane reports
    // the smallest invalid index it owns and the warp takes the minimum, which
    // is the first invalid overall because every position past it is invalid.
    int local_first = width;
    for (int i = lane; i < width; i += warpSize) {
      if (g_ng[i] == k_invalid_id) {
        local_first = i;
        break;
      }
    }
#pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      const int other = shfl_down(local_first, off, warpSize);
      if (other < local_first) {
        local_first = other;
      }
    }
    const int count = shfl_bcast(local_first, 0, warpSize);

    for (int i = lane; i < count; i += warpSize) {
      s_ng[i] = g_ng[i];
    }
    sync_warp();

    // Level 14: sort the whole warp instead of lane 0.
    //
    // This sort exists only to make equal IDs adjacent for the dedup below, so
    // the relative order of equal keys does not matter and the missing
    // tie-break in warp_bitonic_sort is irrelevant here. Padding uses
    // k_invalid_id, which is the maximum value, so the padding sorts to the end
    // and the valid entries stay in [0, count).
    bool sorted_by_warp = false;
    if constexpr (kBitonicSort) {
      int n_pad = 1;
      while (n_pad < count) {
        n_pad <<= 1;
      }
      if (n_pad <= k_native_warp_size * k_bitonic_m && n_pad <= s_capacity) {
        for (int i = count + lane; i < n_pad; i += warpSize) {
          s_ng[i] = k_invalid_id;
        }
        sync_warp();
        warp_bitonic_sort<IDType, void, k_bitonic_m, k_native_warp_size>(
            s_ng, nullptr, n_pad);
        sync_warp();
        sorted_by_warp = true;
      }
    }

    int unique_count = 0;
    if (lane == 0) {
      if (!sorted_by_warp) {
        for (int i = 0; i < count - 1; ++i) {
          for (int j = i + 1; j < count; ++j) {
            if (s_ng[i] > s_ng[j]) {
              swap_values(s_ng[i], s_ng[j]);
            }
          }
        }
      }
      for (int i = 0; i < count; ++i) {
        if (i == 0 || s_ng[i] != s_ng[i - 1]) {
          s_ng[unique_count] = s_ng[i];
          ++unique_count;
        }
      }

      // The shuffle is not cosmetic. The local join samples from these lists,
      // which is what rho controls, so the order decides which neighbours get
      // used. Leaving them sorted by ID biases the sampling and changes the
      // graph. Same seed and same sequence as the original kernel.
      uint64_t rnd_state = static_cast<uint64_t>(sid);
      for (int i = unique_count - 1; i > 0; --i) {
        const int j = static_cast<int>(lcg_rand(rnd_state, i + 1));
        swap_values(s_ng[i], s_ng[j]);
      }

    }
    unique_count = shfl_bcast(unique_count, 0, warpSize);
    sync_warp();

    // Only [0, unique_count) changed, and the stopper marks the end. Anything
    // past it is never read, exactly as in the original kernel.
    for (int i = lane; i < unique_count; i += warpSize) {
      g_ng[i] = s_ng[i];
    }
    if (lane == 0 && unique_count < width) {
      g_ng[unique_count] = k_invalid_id;
    }
    sync_warp();
  }
}

template <typename IDType>
__global__ void remove_duplicate_neighbors_kernel(matrix_view<IDType> ng_ids) {
  const size_t sid      = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n_points = ng_ids.n_rows();
  if (sid >= n_points) {
    return;
  }
  static constexpr auto k_invalid_id = std::numeric_limits<IDType>::max();
  IDType*               ngs          = ng_ids(sid);

  // Count valid neighbors
  int count = 0;
  for (int i = 0; i < ng_ids.n_cols() && ngs[i] != k_invalid_id; ++i) {
    ++count;
  }

  // Sort neighbors
  for (int i = 0; i < count - 1; ++i) {
    for (int j = i + 1; j < count; ++j) {
      if (ngs[i] > ngs[j]) {
        // Swap
        std::swap(ngs[i], ngs[j]);
      }
    }
  }

  // Remove duplicates
  int unique_count = 0;
  for (int i = 0; i < count; ++i) {
    if (i == 0 || ngs[i] != ngs[i - 1]) {
      ngs[unique_count] = ngs[i];
      ++unique_count;
    }
  }
  assert(unique_count <= count);

#ifndef NDEBUG
  // Check duplicates
  for (int i = 0; i < unique_count - 1; ++i) {
    for (int j = i + 1; j < unique_count; ++j) {
      assert(ngs[i] != ngs[j]);
    }
  }
#endif

  // Randomly shuffle
  uint64_t rnd_state = static_cast<uint64_t>(sid);
  for (int i = unique_count - 1; i > 0; --i) {
    const int j = static_cast<int>(lcg_rand(rnd_state, i + 1));
    std::swap(ngs[i], ngs[j]);
  }

  // Add stopper
  if (unique_count < ng_ids.n_cols()) {
    ngs[unique_count] = k_invalid_id;
  }
}

template <typename IDType, typename DistType>
SALTATLAS_HD_GLOBAL void init_neighbor_check_data(
    matrix_view<IDType> candidates_ids, matrix_view<DistType> candidates_dists,
    span<int> candidates_counts) {
  (void)candidates_ids;
  (void)candidates_dists;
  const size_t g_tid =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
      static_cast<size_t>(threadIdx.x);
  const size_t g_warp_id = g_tid / warpSize;
  const int    lane_id   = threadIdx.x % warpSize;
  const size_t n_global_warps =
      (static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x)) /
      static_cast<size_t>(warpSize);
  if (n_global_warps == 0) {
    return;
  }

  for (size_t sid = g_warp_id; sid < candidates_counts.size();
       sid += n_global_warps) {
    if (lane_id == 0) {
      candidates_counts[sid] = 0;
    }
  }
}

// A single warp works on a pair of neighbor lists (nbs1 and nbs2) to find
// better neighbors for points in nbs1.
//
// kOptLevel selects cumulative optimizations. 0 is the unmodified baseline;
// each level includes every lower one.
//   1 : team_any() vote in place of the shfl_down chain + broadcast
//   2 : 32-bit induction variable in the distance loop
//   3 : nid1's feature vector staged in shared memory (needs shared_vecs)
//   4 : level 2 plus 128-bit vector loads in the distance loop
//   6 : reverse push on the asymmetric pass, so (old,new) can be dropped
//   7 : level 6 plus the upper-triangle loop on the symmetric pass
//
// kSymmetricPass says whether nbs1 and nbs2 are the same list. The launcher
// knows this and the kernel cannot cheaply detect it.
template <typename IDType, typename FEType, typename DistType, typename DistOp,
          int kOptLevel = 0, bool kSymmetricPass = false>
SALTATLAS_HD_DEVICE inline void check_neighbors_kernel(
    const IDType* const nbs1, const int n_nbs1, const IDType* const nbs2,
    const int n_nbs2, const matrix_view<FEType>& pstore,
    matrix_view<IDType> candidates_ids, matrix_view<DistType> candidates_dists,
    span<int> candidates_counts, matrix_view<IDType> knng_ids,
    matrix_view<DistType> knng_dists, IDType* shared_knng_ids,
    FEType* shared_vecs) {
  static constexpr auto k_invalid_id = std::numeric_limits<IDType>::max();
  // const int    g_tid = static_cast<int>(blockIdx.x * blockDim.x +
  // threadIdx.x);
  const int    lane_id       = threadIdx.x % warpSize;
  const int    team_id       = lane_id / k_team_size;
  const int    tl_lane_id    = lane_id % k_team_size;
  const int    block_team_id = threadIdx.x / k_team_size;
  const int    n_teams       = warpSize / k_team_size;
  const DistOp dist_op{};
  // Depends only on threadIdx.x, so hoist it out of the pair loop below.
  const team_mask_t t_mask = team_mask(k_team_size);
  // One scratch KNNG row per team in this block:
  // [team0 k entries][team1 k entries]...
  IDType* const team_knng_ids =
      shared_knng_ids + static_cast<size_t>(block_team_id) * knng_ids.n_cols();

  // Push each distance to nid2's candidate list as well as nid1's. On the
  // asymmetric pass this replaces the separate (old,new) launch entirely.
  constexpr bool k_reverse_push = (kOptLevel >= 6) && !kSymmetricPass;

  // Each point in nbs1 is processed by a different team of threads
  for (int i = team_id; i < n_nbs1; i += n_teams) {
    const IDType nid1 = nbs1[i];
    if (nid1 == k_invalid_id) {
      continue;
    }
    // Cache nid1's current KNNG IDs (MSB-cleared) in shared memory so we can
    // skip distance calculations for nid2 already present in KNNG.
    const auto* const nid1_knng_ids = knng_ids(nid1);
    for (int j = tl_lane_id; j < knng_ids.n_cols(); j += k_team_size) {
      team_knng_ids[j] = clear_msb(nid1_knng_ids[j]);
    }
    // Skip team-level memory synchronization here
    // because it's still okay even if team_knng_ids is not fully populated when
    // some threads start checking neighbors.

    // Stage nid1's feature vector in shared memory. It is re-read once per
    // candidate in the loop below, so one cooperative copy here replaces
    // n_nbs2 global reads of the same bytes. Unlike team_knng_ids above this
    // does need a team barrier: a partially written vector would produce a
    // wrong distance rather than merely a missed skip.
    const FEType* nid1_vec = pstore(nid1);
    if constexpr (kOptLevel == 3) {
      const int     n_dims = static_cast<int>(pstore.n_cols());
      FEType* const team_vec =
          shared_vecs + static_cast<size_t>(block_team_id) * pstore.n_cols();
      for (int d = tl_lane_id; d < n_dims; d += k_team_size) {
        team_vec[d] = nid1_vec[d];
      }
      team_sync(t_mask);
      nid1_vec = team_vec;
    }

    // All points in nbs2 are processed by the same team, and the team keeps
    // top-k_top_candidates best neighbors among them.
    IDType   best_ids[k_top_candidates];
    DistType best_dists[k_top_candidates];
#pragma unroll
    for (int b = 0; b < k_top_candidates; ++b) {
      best_ids[b]   = k_invalid_id;
      best_dists[b] = std::numeric_limits<DistType>::max();
    }

    for (int j = 0; j < n_nbs2; ++j) {
      const IDType nid2 = nbs2[j];
      if (nid2 == k_invalid_id || nid2 == nid1) {
        continue;
      }

      // Team lanes cooperatively scan the cached nid1's KNNG list to avoid
      // redundant distance calculations for nid2.
      bool l_in_knng = false;
      for (int kk = tl_lane_id; kk < knng_ids.n_cols(); kk += k_team_size) {
        if (team_knng_ids[kk] == nid2) {
          l_in_knng = true;
          break;
        }
      }

      // Reduce to a single "nid2 already in KNNG" flag within the team.
      bool in_knng;
      if constexpr (kOptLevel >= 1) {
        // Step 1: one vote instruction in place of (k_team_size - 1)
        // shfl_down calls plus a broadcast. The result already lands in every
        // lane, so nothing has to be broadcast back.
        in_knng = team_any(l_in_knng, t_mask);
      } else {
        int           in_knng_flag  = l_in_knng ? 1 : 0;
        constexpr int team_lead_lane = 0;
        // Gather the in_knng flag to the team leader lane
#pragma unroll
        for (int kk = 1; kk < k_team_size; ++kk) {
          in_knng_flag |= shfl_down(in_knng_flag, kk, k_team_size);
        }
        // Broadcast the in_knng flag from the team leader lane to all team
        // members
        in_knng_flag = shfl_bcast(in_knng_flag, team_lead_lane, k_team_size);
        in_knng      = (in_knng_flag != 0);
      }
      if (in_knng) {
        // Already connected in current KNNG; skip expensive distance op.
        continue;
      }

      DistType dist =
          dist_op.template team<FEType, k_team_size, (kOptLevel >= 2),
                                (kOptLevel >= 4)>(nid1_vec, pstore(nid2),
                                                  pstore.n_cols());
      // This distance is a candidate for nid2 just as much as for nid1, and on
      // a symmetric or merged pass nobody else will compute it. There is no
      // per-nid2 accumulator in this loop shape, so filter by distance against
      // nid2's current worst neighbour instead of by rank: it approximates the
      // column-minimum a materialised matrix would give, and keeps the
      // fixed-width candidate buffer from filling with entries that could never
      // survive. Assumes each KNNG row is kept in ascending distance order, so
      // the last column is the worst; recall is the check on that.
      //
      // Only the team leader holds a valid dist (see the warning on l2_team).
      if constexpr (k_reverse_push) {
        if (tl_lane_id == 0) {
          const auto* const nid2_dists = knng_dists(nid2);
          if (dist < nid2_dists[knng_dists.n_cols() - 1]) {
            const auto pos_rev = atomicAdd(&candidates_counts[nid2], 1);
            if (pos_rev < candidates_ids.n_cols()) {
              candidates_ids(nid2, pos_rev)   = nid1;
              candidates_dists(nid2, pos_rev) = dist;
            }
          }
        }
      }

      // Only the team leader update the local candidate list since it's lenght
      // is small
      if (tl_lane_id == 0) {
        if constexpr (k_top_candidates == 1) {
          if (dist < best_dists[0]) {
            best_dists[0] = dist;
            best_ids[0]   = nid2;
          }
          continue;
        } else {
          // only team leader lane updates the top candidates to reduce
          // contention.
          int      max_pos  = -1;
          DistType max_dist = best_dists[0];
#pragma unroll
          for (int b = 1; b < k_top_candidates; ++b) {
            if (best_dists[b] > max_dist) {
              max_dist = best_dists[b];
              max_pos  = b;
            }
          }
          if (dist < max_dist) {
            best_dists[max_pos] = dist;
            best_ids[max_pos]   = nid2;
          }
        }
      }
    }
    // Finished finding top candidates for nid1
    // Update the global candidate list with the top candidates using atomic
    // operations.
    if (tl_lane_id == 0) {
      // Broadcast the number of valid candidates to all team members
#pragma unroll
      for (int b = 0; b < k_top_candidates; ++b) {
        if (best_ids[b] == k_invalid_id) {
          break;
        }
        // Jus atomically push candidate data to the shared candidate buffer
        // without team-level synchronization
        const auto cid     = best_ids[b];
        const auto cdist   = best_dists[b];
        const auto pos_add = atomicAdd(&candidates_counts[nid1], 1);
        if (pos_add < candidates_ids.n_cols()) {
          candidates_ids(nid1, pos_add)   = cid;
          candidates_dists(nid1, pos_add) = cdist;
        } else {
          // Global candidate list is full already,
          // just discard this candidate, hoping it will be included in the
          // future if it's a true nearest neighbor.
        }
      }
    }
  }
}

// Calls check_neighbors_with_global_atomic_per_warp() for each point in
// parallel.
//
// kOptLevel is forwarded to check_neighbors_kernel; 0 reproduces the baseline
// exactly.
//
// The body lives in a device function so that two entry points can share it.
// __launch_bounds__ cannot be conditionally absent within a single template,
// and it is not neutral when present: stating (1024, 1) changed register
// allocation measurably at both level 0 and level 4, in opposite directions.
// Keeping the attribute off the ordinary entry point is what preserves level 0
// as the exact baseline.
template <typename IDType, typename FEType, typename DistType, typename DistOp,
          int kOptLevel, bool kSymmetricPass>
SALTATLAS_HD_DEVICE inline void find_new_neighbor_candidates_impl(
    const matrix_view<IDType> nbs1, const matrix_view<IDType> nbs2,
    const matrix_view<FEType> pstore, matrix_view<IDType> candidates_ids,
    matrix_view<DistType> candidates_dists, span<int> candidates_counts,
    matrix_view<IDType> knng_ids, matrix_view<DistType> knng_dists) {
  // Dynamic shared memory holds the per-team KNNG id scratch, followed (at
  // kOptLevel >= 3) by one staged feature vector per team. The launcher sizes
  // the allocation to match.
  extern __shared__ IDType shared_knng_ids[];
  FEType*                  shared_vecs = nullptr;
  if constexpr (kOptLevel == 3) {
    const size_t n_teams_per_block =
        static_cast<size_t>(blockDim.x) / static_cast<size_t>(k_team_size);
    shared_vecs = reinterpret_cast<FEType*>(shared_knng_ids +
                                            n_teams_per_block *
                                                knng_ids.n_cols());
  }
  const size_t             g_tid =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
      static_cast<size_t>(threadIdx.x);
  const size_t g_warp_id = g_tid / warpSize;
  const size_t n_global_warps =
      (static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x)) /
      static_cast<size_t>(warpSize);
  if (n_global_warps == 0) {
    return;
  }

  for (size_t sid = g_warp_id; sid < pstore.n_rows(); sid += n_global_warps) {
    check_neighbors_kernel<IDType, FEType, DistType, DistOp, kOptLevel,
                           kSymmetricPass>(
        nbs1(sid), nbs1.n_cols(), nbs2(sid), nbs2.n_cols(), pstore,
        candidates_ids, candidates_dists, candidates_counts, knng_ids,
        knng_dists, shared_knng_ids, shared_vecs);
  }
}

// Ordinary entry point, levels 0-4. No launch bound, so codegen matches what
// the kernel produced before any of this parameterisation existed.
template <typename IDType, typename FEType, typename DistType, typename DistOp,
          int kOptLevel = 0, bool kSymmetricPass = false>
SALTATLAS_HD_GLOBAL void find_new_neighbor_candidates(
    const matrix_view<IDType> nbs1, const matrix_view<IDType> nbs2,
    const matrix_view<FEType> pstore, matrix_view<IDType> candidates_ids,
    matrix_view<DistType> candidates_dists, span<int> candidates_counts,
    matrix_view<IDType> knng_ids, matrix_view<DistType> knng_dists) {
  find_new_neighbor_candidates_impl<IDType, FEType, DistType, DistOp,
                                    kOptLevel, kSymmetricPass>(
      nbs1, nbs2, pstore, candidates_ids, candidates_dists, candidates_counts,
      knng_ids, knng_dists);
}

// Level 5 entry point: same body, with registers capped so more blocks fit per
// SM. The dominant stall is waiting on L1TEX, which resident warps hide.
template <typename IDType, typename FEType, typename DistType, typename DistOp,
          int kOptLevel, int kMinBlocks, bool kSymmetricPass>
SALTATLAS_HD_GLOBAL SALTATLAS_SOLANET_NND_LAUNCH_CAP(kMinBlocks) void
find_new_neighbor_candidates_capped(
    const matrix_view<IDType> nbs1, const matrix_view<IDType> nbs2,
    const matrix_view<FEType> pstore, matrix_view<IDType> candidates_ids,
    matrix_view<DistType> candidates_dists, span<int> candidates_counts,
    matrix_view<IDType> knng_ids, matrix_view<DistType> knng_dists) {
  find_new_neighbor_candidates_impl<IDType, FEType, DistType, DistOp,
                                    kOptLevel, kSymmetricPass>(
      nbs1, nbs2, pstore, candidates_ids, candidates_dists, candidates_counts,
      knng_ids, knng_dists);
}

// Each thread independently updates its own KNNG list using the candidates in
// the shared buffer.
// Body shared by the plain and register-capped entry points below.
// Dedup the candidates and drop any that are already neighbours.
//
// Split out of update_one_point so the warp-collective path can run the two
// sorts across all lanes and call only these serial phases on lane 0.
// Assumes the candidates are sorted by ID on entry. Returns the new count.
template <typename IDType, typename DistType, bool kHoistDupCheck>
SALTATLAS_HD_DEVICE inline int update_dedup_filter(
    IDType* const candidate_ids, DistType* const candidate_dists,
    int n_candidates, const IDType* const nids, const int k) {
  n_candidates = remove_duplicate_neighbors<IDType, DistType>(
      candidate_ids, candidate_dists, n_candidates);
  // Level 9: drop candidates that are already neighbours, here rather than
  // inside the merge below.
  //
  // The merge re-scans the whole KNNG row for every candidate it examines,
  // which is O((n_candidates + k) * k): about 3k comparisons at k=32 and 12k at
  // k=64, and the dominant cost of this kernel. The candidates are already
  // sorted by ID at this point, so the same question can be answered by walking
  // the KNNG row once and binary-searching the candidates, which is
  // O(k log n_candidates), around 190 comparisons at k=32.
  //
  // This also checks against the whole KNNG row, where the merge only ever
  // scanned the part it had not yet overwritten, so it can drop a candidate the
  // merge would have inserted as a duplicate. It can never keep one the merge
  // would have dropped.
  if constexpr (kHoistDupCheck) {
    // Hits are recorded in a bitmask rather than written into the array. The
    // search needs the candidates to stay sorted by ID, and overwriting a slot
    // with a sentinel breaks that for every search that follows it.
    //
    // The buffer is candidate_width wide, which is 2k at this level, so four
    // 64-bit words cover any k up to 128. The KNNG row is held per thread, so
    // larger k is not a configuration this kernel supports.
    constexpr int      k_dead_words = 4;
    unsigned long long dead[k_dead_words] = {0ull, 0ull, 0ull, 0ull};
    assert(n_candidates <= 64 * k_dead_words &&
           "hoisted duplicate check: candidate buffer wider than the bitmask");

    for (int i = 0; i < k; ++i) {
      const IDType kid = clear_msb(nids[i]);
      int          lo = 0, hi = n_candidates - 1;
      while (lo <= hi) {
        const int    mid = lo + ((hi - lo) >> 1);
        const IDType cid = candidate_ids[mid];
        if (cid == kid) {
          dead[mid >> 6] |= (1ull << (mid & 63));
          break;
        }
        if (cid < kid) {
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
    }

    // Compact in place, preserving ascending ID order.
    int w = 0;
    for (int r = 0; r < n_candidates; ++r) {
      if (((dead[r >> 6] >> (r & 63)) & 1ull) == 0ull) {
        candidate_ids[w]   = candidate_ids[r];
        candidate_dists[w] = candidate_dists[r];
        ++w;
      }
    }
    n_candidates = w;
  }


  return n_candidates;
}

// Merge the candidates into the KNNG row and return how many were installed.
//
// Split out of update_one_point so the warp-collective path can run the two
// sorts across all lanes and call only the serial phases on lane 0. Assumes
// the candidates are sorted by distance on entry.
template <typename IDType, typename FEType, typename DistType,
          bool kHoistDupCheck, bool kSkipCleanResort, bool kMergeResort>
SALTATLAS_HD_DEVICE inline int update_merge_row(
    IDType* const candidate_ids, DistType* const candidate_dists,
    const int n_candidates, IDType* const nids, DistType* const dists,
    const int k) {
  // Merge candidates into KNNG list.
  // Both input KNNG and candidates lists must be sorted by distance (ID
  // breaks ties) and must not have duplicates.
  int knng_idx    = 0;
  int c_idx       = 0;
  int knng_tail   = k - 1;
  int l_n_updates = 0;
  while (c_idx < n_candidates && knng_idx <= knng_tail) {
    // const auto knng_id   = clear_msb(nids[knng_idx]);
    const auto knng_dist = dists[knng_idx];
    const auto cid       = candidate_ids[c_idx];
    const auto cdist     = candidate_dists[c_idx];

    // At level 9 this was answered before the loop; see the hoisted check above.
    if constexpr (!kHoistDupCheck) {
      // TODO: use bit-map hash table to reduce the for loop check?
      bool duplicate = false;
      for (int i = 0; i <= knng_tail; ++i) {
        if (clear_msb(nids[i]) == cid) {
          duplicate = true;
          break;
        }
      }
      if (duplicate) {
        // candidate is already in KNNG list
        ++c_idx;
        continue;
      }
    }

    if (!nearly_equal(cdist, knng_dist) && cdist < knng_dist) {
      // Candidate is better than the current neighbor
      // Insert as new neighbor with MSB set
      nids[knng_tail]  = set_msb<IDType>(cid);
      dists[knng_tail] = cdist;
      ++l_n_updates;
      --knng_tail;
      ++c_idx;
    } else {
      // Candidate is worse than or equal to the current neighbor, move to the
      // next neighbor in KNNG list
      ++knng_idx;
    }
  }
    if constexpr (kMergeResort) {
      // Level 11: the merge already leaves two sorted runs, so combine them in
      // O(k) instead of re-sorting the whole row in O(k^2).
      //
      //   [0 .. knng_tail]      untouched prefix, still ascending
      //   [knng_tail+1 .. k-1]  the insertions, DESCENDING: the best candidate
      //                         was written first, at the highest index, and
      //                         each later one at a lower index
      //
      // The full re-sort was 34% of this kernel at three iterations. Level 10
      // tried to skip it when nothing was written, which bought nothing: with
      // one thread per point, a warp takes the branch if any of its 32 threads
      // has an update, so the test almost never skips. This wins per thread
      // regardless of what the rest of the warp is doing.
      constexpr int k_tmp_max = 64;
      const int     n_ins     = k - 1 - knng_tail;
      if (n_ins > 0 && n_ins <= k_tmp_max) {
        IDType   t_ids[k_tmp_max];
        DistType t_dists[k_tmp_max];
        // Reverse the insertions into the temp so both runs run ascending.
        for (int j = 0; j < n_ins; ++j) {
          t_ids[j]   = nids[k - 1 - j];
          t_dists[j] = dists[k - 1 - j];
        }
        // Merge from the back. w is always >= i, so writing at w never
        // clobbers a prefix element still to be read.
        int i = knng_tail;
        int j = n_ins - 1;
        int w = k - 1;
        while (j >= 0) {
          bool take_prefix = false;
          if (i >= 0) {
            // Match sort_neighbors_single_thread: ascending distance, ties
            // broken by ascending ID.
            take_prefix = nearly_equal(dists[i], t_dists[j])
                              ? (clear_msb(nids[i]) > clear_msb(t_ids[j]))
                              : (dists[i] > t_dists[j]);
          }
          if (take_prefix) {
            nids[w]  = nids[i];
            dists[w] = dists[i];
            --i;
          } else {
            nids[w]  = t_ids[j];
            dists[w] = t_dists[j];
            --j;
          }
          --w;
        }
      } else if (n_ins > k_tmp_max) {
        sort_neighbors_single_thread(nids, dists, k);
      }
    } else if (!kSkipCleanResort || l_n_updates > 0) {
      // Level 10, kept as a documented negative result: skipping the re-sort
      // for an untouched row measured neutral, because the branch is per
      // thread and the warp is not.
      sort_neighbors_single_thread(nids, dists, k);
    }  return l_n_updates;
}


template <typename IDType, typename FEType, typename DistType,
          bool kHoistDupCheck = false, bool kSkipCleanResort = false,
          bool kMergeResort = false>
SALTATLAS_HD_DEVICE inline int update_one_point(
    IDType* const candidate_ids, DistType* const candidate_dists,
    int n_candidates, IDType* const nids, DistType* const dists, const int k) {
  // Sort candidates by ID to remove duplicate IDs
  // Sorting by distance may not adjacent duplicate IDs due to distance value's
  // float precision problem.
    sort_neighbors_single_thread(candidate_ids, candidate_dists, n_candidates,
                                 false);  n_candidates = update_dedup_filter<IDType, DistType, kHoistDupCheck>(
      candidate_ids, candidate_dists, n_candidates, nids, k);

    sort_neighbors_single_thread(candidate_ids, candidate_dists, n_candidates,
                                 true);
  return update_merge_row<IDType, FEType, DistType, kHoistDupCheck,
                         kSkipCleanResort, kMergeResort>(
      candidate_ids, candidate_dists, n_candidates, nids, dists, k);
}

// One thread per point. The original launch shape, unchanged.
template <typename IDType, typename FEType, typename DistType,
          bool kHoistDupCheck = false, bool kSkipCleanResort = false,
          bool kMergeResort = false>
SALTATLAS_HD_DEVICE inline void update_knng_with_candidates_impl(
    matrix_view<IDType>   candidates_ids_table,
    matrix_view<DistType> candidates_dists_table, span<int> candidates_counts,
    matrix_view<IDType> knng_ids, matrix_view<DistType> knng_dists,
    size_t* n_updates_block) {
  const size_t g_tid =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
      static_cast<size_t>(threadIdx.x);
  const IDType sid = static_cast<int>(g_tid);
  if (sid >= knng_ids.n_rows()) {
    return;
  }

  auto* const nids  = knng_ids(sid);
  auto* const dists = knng_dists(sid);
  const int   k     = knng_ids.n_cols();
  const int   n_candidates =
      std::min<int>(candidates_counts[sid], candidates_ids_table.n_cols());

  const int l_n_updates =
      update_one_point<IDType, FEType, DistType, kHoistDupCheck,
                       kSkipCleanResort, kMergeResort>(
          candidates_ids_table(sid), candidates_dists_table(sid), n_candidates,
          nids, dists, k);

  if (l_n_updates > 0) {
    atomic_add_u64(&n_updates_block[blockIdx.x],
                   static_cast<size_t>(l_n_updates));
  }

#ifndef NDEBUG
  sort_neighbors_single_thread(nids, dists, k, false);
  if (remove_duplicate_neighbors(nids, dists, k) != k) {
    printf("Duplicates found in KNNG list of point %d after update.\n", sid);
    for (int i = 0; i < k; ++i) {
      printf("%d  Neighbor %d: ID %u, distance %f\n", sid, i,
             clear_msb(nids[i]), dists[i]);
    }
    assert(false && "Duplicates found in KNNG list after update.");
  }
  sort_neighbors_single_thread(nids, dists, k, true);
#endif

#ifndef NDEBUG
  // Check that KNNG list is still sorted after the update
  for (int i = 0; i < k - 1; ++i) {
    if (dists[i] < dists[i + 1]) {
      continue;  // good
    }

    if (nearly_equal(dists[i], dists[i + 1])) {
      if (clear_msb(nids[i]) < clear_msb(nids[i + 1])) {
        continue;  // good
      }
      printf(
          "Distance at position %d is equal to distance at position %d but "
          "ID is greater for point %d. ID %u and %u (distance %f and %f)\n",
          i, i + 1, sid, clear_msb(nids[i]), clear_msb(nids[i + 1]), dists[i],
          dists[i + 1]);
    } else if (dists[i] > dists[i + 1]) {
      printf(
          "Distance at position %d is greater than distance at position %d "
          "for point %d. Distance %f and %f\n",
          i, i + 1, sid, dists[i], dists[i + 1]);
    }

    assert(dists[i] < dists[i + 1] ||
           (nearly_equal(dists[i], dists[i + 1]) &&
            clear_msb(nids[i]) < clear_msb(nids[i + 1])));
  }

  // Check that there are no duplicates in the KNNG list
  for (int i = 0; i < k - 1; ++i) {
    for (int j = i + 1; j < k; ++j) {
      if (clear_msb(nids[i]) == clear_msb(nids[j])) {
        printf(
            "Duplicate neighbor ID %u found in KNNG list of point %d at "
            "positions %d and %d. Distance %f and %f\n",
            clear_msb(nids[i]), sid, i, j, dists[i], dists[j]);
      }
      assert(clear_msb(nids[i]) != clear_msb(nids[j]));
    }
  }
#endif
}

// One warp per point (level 12).
//
// Stage 1 of the warp-collective restructuring: the launch shape and the
// shared-memory staging are in place, but the body between them is still the
// serial algorithm run by lane 0. This is EXPECTED TO BE SLOWER than level 11,
// the same serial work with a thirty-second of the point-level parallelism. It
// exists so that when the sorts and the merge move to the whole warp in the
// stages after this one, a change in behaviour can be attributed to the
// algorithm rather than to the data movement.
//
// The staging itself is already warp-parallel, and that is the part that
// motivates the whole exercise. One thread per point means thread sid reads row
// sid, so 32 consecutive threads touch 32 addresses a full row apart and every
// access is its own transaction. Here 32 lanes read consecutive elements of one
// row, which coalesces.
//
// Shared memory per warp: (candidate_width + k) ids and the same number of
// distances. The launcher sizes the dynamic allocation to match.
template <typename IDType, typename FEType, typename DistType,
          bool kHoistDupCheck = false, bool kMergeResort = false,
          bool kWarpSort = false>
SALTATLAS_HD_DEVICE inline void update_knng_with_candidates_warp_impl(
    matrix_view<IDType>   candidates_ids_table,
    matrix_view<DistType> candidates_dists_table, span<int> candidates_counts,
    matrix_view<IDType> knng_ids, matrix_view<DistType> knng_dists,
    size_t* n_updates_block, size_t n_update_slots) {
  extern __shared__ char smem_upd[];

  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  const int warps_per_block = static_cast<int>(blockDim.x) / warpSize;
  const int warp_in_block   = static_cast<int>(threadIdx.x) / warpSize;

  const int cw = static_cast<int>(candidates_ids_table.n_cols());
  const int k  = static_cast<int>(knng_ids.n_cols());
  // Level 15 pads the candidate run to a power of two for the bitonic network,
  // so the candidate region is allocated at that size.
  int cap = 1;
  while (cap < cw) {
    cap <<= 1;
  }
  const int cand_slots = kWarpSort ? cap : cw;
  const int n_slots    = cand_slots + k;
  constexpr int k_sort_m = 8;

  IDType* const   s_id_base = reinterpret_cast<IDType*>(smem_upd);
  DistType* const s_ds_base = reinterpret_cast<DistType*>(
      s_id_base + static_cast<size_t>(warps_per_block) * n_slots);

  IDType* const   s_cid   = s_id_base + warp_in_block * n_slots;
  IDType* const   s_nid   = s_cid + cand_slots;
  DistType* const s_cdist = s_ds_base + warp_in_block * n_slots;
  DistType* const s_ndist = s_cdist + cand_slots;

  const size_t g_warp =
      (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warpSize;
  const size_t n_warps =
      (static_cast<size_t>(gridDim.x) * blockDim.x) / warpSize;
  if (n_warps == 0) {
    return;
  }

  for (size_t sid = g_warp; sid < knng_ids.n_rows(); sid += n_warps) {
    const int n_cand =
        std::min<int>(candidates_counts[sid], cw);

    const IDType* const   g_cid   = candidates_ids_table(sid);
    const DistType* const g_cdist = candidates_dists_table(sid);
    IDType* const         g_nid   = knng_ids(sid);
    DistType* const       g_ndist = knng_dists(sid);

    for (int i = lane; i < n_cand; i += warpSize) {
      s_cid[i]   = g_cid[i];
      s_cdist[i] = g_cdist[i];
    }
    for (int i = lane; i < k; i += warpSize) {
      s_nid[i]   = g_nid[i];
      s_ndist[i] = g_ndist[i];
    }
    sync_warp();

    int l_n_updates = 0;
    if constexpr (kWarpSort) {
      // Level 15: the two candidate sorts run across the whole warp; only the
      // dedup, the duplicate filter and the merge stay on lane 0.
      int n_pad = 1;
      while (n_pad < n_cand) {
        n_pad <<= 1;
      }
      const bool fits = (n_pad <= k_native_warp_size * k_sort_m) &&
                        (n_pad <= cand_slots);
      if (fits) {
        // Sort by ID, ties by distance, so the dedup below deterministically
        // keeps the nearest of each duplicate group. The serial path never did
        // that: it sorted by ID alone and kept whichever landed first.
        for (int i = n_cand + lane; i < n_pad; i += warpSize) {
          s_cid[i]   = std::numeric_limits<IDType>::max();
          s_cdist[i] = std::numeric_limits<DistType>::max();
        }
        sync_warp();
        warp_bitonic_sort<IDType, DistType, k_sort_m, k_native_warp_size, true>(
            s_cid, s_cdist, n_pad);
        sync_warp();

        int n_kept = 0;
        if (lane == 0) {
          n_kept = update_dedup_filter<IDType, DistType, kHoistDupCheck>(
              s_cid, s_cdist, n_cand, s_nid, k);
        }
        n_kept = shfl_bcast(n_kept, 0, warpSize);
        sync_warp();

        // Sort by distance, ties by ID: the ordering the KNNG row requires and
        // the debug build asserts on.
        int m_pad = 1;
        while (m_pad < n_kept) {
          m_pad <<= 1;
        }
        for (int i = n_kept + lane; i < m_pad; i += warpSize) {
          s_cdist[i] = std::numeric_limits<DistType>::max();
          s_cid[i]   = std::numeric_limits<IDType>::max();
        }
        sync_warp();
        warp_bitonic_sort<DistType, IDType, k_sort_m, k_native_warp_size, true>(
            s_cdist, s_cid, m_pad);
        sync_warp();

        if (lane == 0) {
          l_n_updates =
              update_merge_row<IDType, FEType, DistType, kHoistDupCheck,
                               false, kMergeResort>(s_cid, s_cdist, n_kept,
                                                    s_nid, s_ndist, k);
        }
      } else if (lane == 0) {
        l_n_updates = update_one_point<IDType, FEType, DistType,
                                       kHoistDupCheck, false, kMergeResort>(
            s_cid, s_cdist, n_cand, s_nid, s_ndist, k);
      }
    } else if (lane == 0) {
      l_n_updates = update_one_point<IDType, FEType, DistType,
                                     kHoistDupCheck, false, kMergeResort>(
          s_cid, s_cdist, n_cand, s_nid, s_ndist, k);
    }
    sync_warp();

    for (int i = lane; i < k; i += warpSize) {
      g_nid[i]   = s_nid[i];
      g_ndist[i] = s_ndist[i];
    }

    // n_updates_block has one slot per block of the ONE-THREAD-PER-POINT grid,
    // which is 32x smaller than grid_warp_points. Indexing it by blockIdx.x
    // here would write past the end for most blocks, undercount the updates,
    // shrink delta and stop the algorithm early. Fold into a slot that is
    // always in range instead; the driver only ever sums the array.
    if (lane == 0 && l_n_updates > 0) {
      atomic_add_u64(&n_updates_block[blockIdx.x % n_update_slots],
                     static_cast<size_t>(l_n_updates));
    }
    sync_warp();
  }
}

template <typename IDType, typename FEType, typename DistType,
          bool kHoistDupCheck = false, bool kMergeResort = false,
          bool kWarpSort = false>
SALTATLAS_HD_GLOBAL void update_knng_with_candidates_warp(
    matrix_view<IDType>   candidates_ids_table,
    matrix_view<DistType> candidates_dists_table, span<int> candidates_counts,
    matrix_view<IDType> knng_ids, matrix_view<DistType> knng_dists,
    size_t* n_updates_block, size_t n_update_slots) {
  update_knng_with_candidates_warp_impl<IDType, FEType, DistType,
                                        kHoistDupCheck, kMergeResort,
                                        kWarpSort>(
      candidates_ids_table, candidates_dists_table, candidates_counts, knng_ids,
      knng_dists, n_updates_block, n_update_slots);
}

// Ordinary entry point. No launch bound, so codegen matches the original.
template <typename IDType, typename FEType, typename DistType,
          bool kHoistDupCheck = false, bool kSkipCleanResort = false,
          bool kMergeResort = false>
SALTATLAS_HD_GLOBAL void update_knng_with_candidates(
    matrix_view<IDType>   candidates_ids_table,
    matrix_view<DistType> candidates_dists_table, span<int> candidates_counts,
    matrix_view<IDType> knng_ids, matrix_view<DistType> knng_dists,
    size_t* n_updates_block) {
  update_knng_with_candidates_impl<IDType, FEType, DistType, kHoistDupCheck,
                                   kSkipCleanResort, kMergeResort>(
      candidates_ids_table, candidates_dists_table, candidates_counts, knng_ids,
      knng_dists, n_updates_block);
}

/// \brief Set MSB of neighbors in the range [k_begin, k_end) as new neighbors.
/// (No default arguments: CUDA does not allow them on __global__ functions.)
template <typename IDType>
SALTATLAS_HD_GLOBAL void set_msbs(matrix_view<IDType> knng_ids,
                                  const int k_begin, const int k_end) {
  const size_t sid      = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n_points = knng_ids.n_rows();
  const size_t k        = knng_ids.n_cols();
  if (sid >= n_points) {
    return;
  }

  IDType* ids_row = knng_ids(sid);
  for (size_t i = std::min<int>(k_begin, k); i < std::min<int>(k_end, k); ++i) {
    ids_row[i] = set_msb(ids_row[i]);
  }
}

template <typename IDType>
SALTATLAS_HD_GLOBAL void clear_msbs(matrix_view<IDType> knng_ids) {
  const size_t sid      = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n_points = knng_ids.n_rows();
  const size_t k        = knng_ids.n_cols();
  if (sid >= n_points) {
    return;
  }

  IDType* ids_row = knng_ids(sid);
  for (size_t i = 0; i < k; ++i) {
    ids_row[i] = clear_msb(ids_row[i]);
  }
}

template <typename IDType, typename FEType, typename DistType, typename DistOp>
void build_index_main_loop(
    const matrix_view<FEType>& pstore, const int k, const int p_new,
    const int p_old, const float delta, const int max_iterations,
    const hipDeviceProp_t& device_prop, const dim3& grid_points,
    const dim3& grid_warp_points, const dim3& block, const size_t n_blocks,
    matrix_view<IDType> knng_ids, matrix_view<DistType> knng_dists,
    matrix_view<IDType> old_ng, hip_unique_ptr<int>& old_counts,
    matrix_view<IDType> new_ng, hip_unique_ptr<int>& new_counts,
    hip_unique_ptr<int>& old_counts_wk, hip_unique_ptr<int>& new_counts_wk) {
  const size_t n_points = pstore.n_rows();
  // Read here as well as further down, because the candidate buffers are
  // allocated before that declaration is in scope.
  static const int nnd_opt_level_for_alloc = [] {
    const char* const env = std::getenv("SALTATLAS_SOLANET_NND_OPT");
    return (env != nullptr) ? std::atoi(env) : 0;
  }();
  const int candidate_width =
      (nnd_opt_level_for_alloc >= 6)
          ? (k * SALTATLAS_SOLANET_APU_NND_CANDIDATE_WIDTH_NUM) /
                SALTATLAS_SOLANET_APU_NND_CANDIDATE_WIDTH_DEN
          : k;
  matrix<IDType>   candidate_ids(n_points, candidate_width);
  matrix<DistType> candidate_dists(n_points, candidate_width);
  auto             candidate_counts_buf = make_hip_array<int>(n_points);
  auto candidate_counts = span<int>(candidate_counts_buf.get(), n_points);
  auto n_updates_block  = make_hip_array<size_t>(n_blocks);

  int super_step_no = 0;
  while (super_step_no < max_iterations) {
    spdlog::trace("Super step: {}", super_step_no);

    spdlog::trace("Prepare new and old neighbor lists");
    rec_time().start("set_new_old");
    hipLaunchKernelGGL((set_new_old<IDType>), grid_points, block, 0, nullptr,
                       knng_ids, p_new, p_old, new_ng,
                       span<int>(new_counts.get(), n_points), old_ng,
                       span<int>(old_counts.get(), n_points));
    SALTATLAS_HIP_CHECK(hipGetLastError());
    SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
    rec_time().stop();  // set_new_old
#ifdef SALTATLAS_SOLANET_APU_NND_PROFILE
    std::cout << "#of new neighbors (pid = 0): " << new_counts(0, 0)
              << std::endl;
    std::cout << "#of old neighbors (pid = 0): " << old_counts(0, 0)
              << std::endl;
#endif

    spdlog::trace("Add reverse neighbors old");
    rec_time().start("add_rev_old");
    SALTATLAS_HIP_CHECK(hipMemcpy(old_counts_wk.get(), old_counts.get(),
                                  n_points * sizeof(int),
                                  hipMemcpyDeviceToDevice));
    hipLaunchKernelGGL((add_reverse_neighbors<IDType>), grid_points, block, 0,
                       nullptr, old_ng, span<int>(old_counts.get(), n_points),
                       span<int>(old_counts_wk.get(), n_points));
    SALTATLAS_HIP_CHECK(hipGetLastError());
    SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
    rec_time().stop();  // add_reverse_neighbors_old

    spdlog::trace("Remove duplicate neighbors old");
    rec_time().start("remove_dup_old");
    // nnd_opt_level is declared further down; nnd_opt_level_for_alloc reads the
    // same variable and is in scope here.
    if (nnd_opt_level_for_alloc >= 13) {
      const size_t dd_warps = static_cast<size_t>(block.x) / device_prop.warpSize;
      // Rounded up to a power of two so level 14 can pad the run it sorts.
      size_t dd_cap = 1;
      while (dd_cap < old_ng.n_cols()) {
        dd_cap <<= 1;
      }
      const size_t dd_bytes = dd_warps * dd_cap * sizeof(IDType);
      if (nnd_opt_level_for_alloc >= 14) {
        hipLaunchKernelGGL(
            (remove_duplicate_neighbors_warp_kernel<IDType, true>),
            grid_warp_points, block, dd_bytes, nullptr, old_ng);
      } else {
        hipLaunchKernelGGL((remove_duplicate_neighbors_warp_kernel<IDType>),
                           grid_warp_points, block, dd_bytes, nullptr, old_ng);
      }
    } else {
      hipLaunchKernelGGL((remove_duplicate_neighbors_kernel<IDType>),
                         grid_points, block, 0, nullptr, old_ng);
    }
    SALTATLAS_HIP_CHECK(hipGetLastError());
    // SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
    rec_time().stop();  // remove_dup_old
#ifdef SALTATLAS_SOLANET_APU_NND_PROFILE
    std::cout << "#of old neighbors after removing dups (pid = 0): "
              << old_counts(0, 0) << std::endl;
#endif

    spdlog::trace("Add reverse neighbors new");
    rec_time().start("add_rev_new");
    SALTATLAS_HIP_CHECK(hipMemcpy(new_counts_wk.get(), new_counts.get(),
                                  n_points * sizeof(int),
                                  hipMemcpyDeviceToDevice));
    hipLaunchKernelGGL((add_reverse_neighbors<IDType>), grid_points, block, 0,
                       nullptr, new_ng, span<int>(new_counts.get(), n_points),
                       span<int>(new_counts_wk.get(), n_points));
    SALTATLAS_HIP_CHECK(hipGetLastError());
    SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
    rec_time().stop();  // add_reverse_neighbors_new

    spdlog::trace("Remove duplicate neighbors new");
    rec_time().start("remove_dup_new");
    // nnd_opt_level is declared further down; nnd_opt_level_for_alloc reads the
    // same variable and is in scope here.
    if (nnd_opt_level_for_alloc >= 13) {
      const size_t dd_warps = static_cast<size_t>(block.x) / device_prop.warpSize;
      // Rounded up to a power of two so level 14 can pad the run it sorts.
      size_t dd_cap = 1;
      while (dd_cap < new_ng.n_cols()) {
        dd_cap <<= 1;
      }
      const size_t dd_bytes = dd_warps * dd_cap * sizeof(IDType);
      if (nnd_opt_level_for_alloc >= 14) {
        hipLaunchKernelGGL(
            (remove_duplicate_neighbors_warp_kernel<IDType, true>),
            grid_warp_points, block, dd_bytes, nullptr, new_ng);
      } else {
        hipLaunchKernelGGL((remove_duplicate_neighbors_warp_kernel<IDType>),
                           grid_warp_points, block, dd_bytes, nullptr, new_ng);
      }
    } else {
      hipLaunchKernelGGL((remove_duplicate_neighbors_kernel<IDType>),
                         grid_points, block, 0, nullptr, new_ng);
    }
    SALTATLAS_HIP_CHECK(hipGetLastError());
    SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
    rec_time().stop();  // remove_dup_new
#ifdef SALTATLAS_SOLANET_APU_NND_PROFILE
    std::cout << "#of new neighbors after removing dups (pid = 0): "
              << new_counts(0, 0) << std::endl;
#endif

    spdlog::trace("Neighbor checks and KNNG updates");
    SALTATLAS_HIP_CHECK(
        hipMemset(n_updates_block.get(), 0, n_blocks * sizeof(size_t)));

    /// Optimization level for the neighbor-check kernel, selected at run time
    /// so that one binary can reproduce the baseline and every optimized
    /// variant without a rebuild. 0 (the default) is the unmodified baseline.
    ///   1 : team_any() vote in place of the shfl_down chain + broadcast
    ///   2 : 32-bit induction variable in the distance loop
    ///   3 : stage nid1's feature vector in shared memory. Measured slower
    ///       than level 2 (shared and L1 are the same unit, so relocating
    ///       loads does not relieve it); kept only for reference. NOT
    ///       included in level 4.
    ///   4 : level 2 plus 128-bit vector loads in the distance loop
    ///   5 : level 4 plus a register cap, trading spills for occupancy
    ///       (a no-op on AMD; see SALTATLAS_SOLANET_NND_LAUNCH_CAP)
    ///   6 : reverse push on the (new, old) pass; (old, new) is not launched
    ///   7 : level 6 plus the upper-triangle loop on the (new, new) pass
    ///       (measured 2x slower than level 6; kept as a negative result)
    ///   8 : level 6 plus a register cap on update_knng_with_candidates
    ///       (measured neutral; not inherited by level 9)
    ///   9 : level 6 with the duplicate check hoisted out of the merge in
    ///       update_knng_with_candidates
    static const int nnd_opt_level = [] {
      const char* const env = std::getenv("SALTATLAS_SOLANET_NND_OPT");
      return (env != nullptr) ? std::atoi(env) : 0;
    }();

    // The update launch, shared by neighbor_checker and, at level 16, by
    // fused_neighbor_checker.
    auto launch_update_knng = [&]() {
      rec_time().start("knng_updates");
      // Level 9 hoists the duplicate check out of the merge; it is otherwise
      // level 6's update kernel. Level 8's register cap is not inherited,
      // having measured neutral.
      if (nnd_opt_level >= 12) {
        // Level 12: one warp per point. grid_warp_points is the same grid the
        // neighbour check uses. Shared memory holds, per warp, the candidate
        // buffer and the KNNG row as ids then distances.
        const size_t upd_warps_per_block =
            static_cast<size_t>(block.x) / device_prop.warpSize;
        size_t upd_cap = 1;
        while (upd_cap < static_cast<size_t>(candidate_width)) {
          upd_cap <<= 1;
        }
        const size_t upd_slots =
            (nnd_opt_level >= 15 ? upd_cap
                                 : static_cast<size_t>(candidate_width)) +
            static_cast<size_t>(k);
        const size_t upd_shared_bytes =
            upd_warps_per_block * upd_slots *
            (sizeof(IDType) + sizeof(DistType));
        if (nnd_opt_level >= 15) {
          hipLaunchKernelGGL(
              (update_knng_with_candidates_warp<IDType, FEType, DistType, true,
                                                true, true>),
              grid_warp_points, block, upd_shared_bytes, nullptr,
              candidate_ids.get_view(), candidate_dists.get_view(),
              candidate_counts, knng_ids, knng_dists, n_updates_block.get(),
              n_blocks);
        } else {
          hipLaunchKernelGGL(
              (update_knng_with_candidates_warp<IDType, FEType, DistType, true,
                                                true, false>),
              grid_warp_points, block, upd_shared_bytes, nullptr,
              candidate_ids.get_view(), candidate_dists.get_view(),
              candidate_counts, knng_ids, knng_dists, n_updates_block.get(),
              n_blocks);
        }
      } else if (nnd_opt_level >= 11) {
        hipLaunchKernelGGL(
            (update_knng_with_candidates<IDType, FEType, DistType, true, false,
                                         true>),
            grid_points, block, 0, nullptr, candidate_ids.get_view(),
            candidate_dists.get_view(), candidate_counts, knng_ids, knng_dists,
            n_updates_block.get());
      } else if (nnd_opt_level >= 10) {
        hipLaunchKernelGGL(
            (update_knng_with_candidates<IDType, FEType, DistType, true, true>),
            grid_points, block, 0, nullptr, candidate_ids.get_view(),
            candidate_dists.get_view(), candidate_counts, knng_ids, knng_dists,
            n_updates_block.get());
      } else if (nnd_opt_level >= 9) {
        hipLaunchKernelGGL(
            (update_knng_with_candidates<IDType, FEType, DistType, true>),
            grid_points, block, 0, nullptr, candidate_ids.get_view(),
            candidate_dists.get_view(), candidate_counts, knng_ids, knng_dists,
            n_updates_block.get());
      } else {
        hipLaunchKernelGGL(
            (update_knng_with_candidates<IDType, FEType, DistType>),
            grid_points, block, 0, nullptr, candidate_ids.get_view(),
            candidate_dists.get_view(), candidate_counts, knng_ids, knng_dists,
            n_updates_block.get());
      }
      SALTATLAS_HIP_CHECK(hipGetLastError());
      SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
      rec_time().stop();  // knng_updates
    };

    /// Perform neighbor checks for the given pairs of neighbor lists and update
    /// KNNG
    auto neighbor_checker = [&](const matrix_view<IDType>& nbs1,
                                const matrix_view<IDType>& nbs2,
                                const bool                 symmetric_pass) {
      const size_t n_teams_per_block = block.x / k_team_size;
      size_t       ck_shared_bytes =
          n_teams_per_block * static_cast<size_t>(k) * sizeof(IDType);
      if (ck_shared_bytes > device_prop.sharedMemPerBlock) {
        throw std::runtime_error(
            "Neighbor check shared memory exceeds device limit. Reduce k or "
            "increase team size.");
      }

      // Level 3 stages one feature vector per team alongside the id scratch.
      // High-dimensional data can push that past the per-block limit, in which
      // case fall back to level 2 rather than failing: the staging is an
      // optimization, not a requirement.
      int effective_opt = nnd_opt_level;
      if (effective_opt == 3) {
        const size_t vec_bytes =
            n_teams_per_block * pstore.n_cols() * sizeof(FEType);
        if (ck_shared_bytes + vec_bytes <= device_prop.sharedMemPerBlock) {
          ck_shared_bytes += vec_bytes;
        } else {
          effective_opt = 2;
        }
      }

      rec_time().start("init neighbor checks");
      SALTATLAS_HIP_CHECK(
          hipMemset(candidate_counts_buf.get(), 0, n_points * sizeof(int)));
      SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
      rec_time().stop();  // init neighbor checks

      rec_time().start("neighbor_checks");
      // One instantiation per optimization level. Keeping them as distinct
      // template arguments means level 0 compiles to exactly the original
      // kernel, so the baseline stays available for comparison.
#define SALTATLAS_LAUNCH_NEIGHBOR_CHECK(KERNEL, LEVEL)                    \
  do {                                                                    \
    if (symmetric_pass) {                                                 \
      hipLaunchKernelGGL(                                                 \
          (KERNEL<IDType, FEType, DistType, DistOp, (LEVEL), true>),      \
          grid_warp_points, block, ck_shared_bytes, nullptr, nbs1, nbs2,  \
          pstore, candidate_ids.get_view(), candidate_dists.get_view(),   \
          candidate_counts, knng_ids, knng_dists);                        \
    } else {                                                              \
      hipLaunchKernelGGL(                                                 \
          (KERNEL<IDType, FEType, DistType, DistOp, (LEVEL), false>),     \
          grid_warp_points, block, ck_shared_bytes, nullptr, nbs1, nbs2,  \
          pstore, candidate_ids.get_view(), candidate_dists.get_view(),   \
          candidate_counts, knng_ids, knng_dists);                        \
    }                                                                     \
  } while (0)

#define SALTATLAS_LAUNCH_NEIGHBOR_CHECK_CAPPED(LEVEL, MIN_BLOCKS)         \
  do {                                                                    \
    if (symmetric_pass) {                                                 \
      hipLaunchKernelGGL(                                                 \
          (find_new_neighbor_candidates_capped<                           \
              IDType, FEType, DistType, DistOp, (LEVEL), (MIN_BLOCKS),    \
              true>),                                                     \
          grid_warp_points, block, ck_shared_bytes, nullptr, nbs1, nbs2,  \
          pstore, candidate_ids.get_view(), candidate_dists.get_view(),   \
          candidate_counts, knng_ids, knng_dists);                        \
    } else {                                                              \
      hipLaunchKernelGGL(                                                 \
          (find_new_neighbor_candidates_capped<                           \
              IDType, FEType, DistType, DistOp, (LEVEL), (MIN_BLOCKS),    \
              false>),                                                    \
          grid_warp_points, block, ck_shared_bytes, nullptr, nbs1, nbs2,  \
          pstore, candidate_ids.get_view(), candidate_dists.get_view(),   \
          candidate_counts, knng_ids, knng_dists);                        \
    }                                                                     \
  } while (0)

      // Level 5 is level 4's body behind the register-capped entry point.
      switch (effective_opt) {
        case 0:
          SALTATLAS_LAUNCH_NEIGHBOR_CHECK(find_new_neighbor_candidates, 0);
          break;
        case 1:
          SALTATLAS_LAUNCH_NEIGHBOR_CHECK(find_new_neighbor_candidates, 1);
          break;
        case 2:
          SALTATLAS_LAUNCH_NEIGHBOR_CHECK(find_new_neighbor_candidates, 2);
          break;
        case 3:
          SALTATLAS_LAUNCH_NEIGHBOR_CHECK(find_new_neighbor_candidates, 3);
          break;
        case 4:
          SALTATLAS_LAUNCH_NEIGHBOR_CHECK(find_new_neighbor_candidates, 4);
          break;
        case 5:
          // Level 5 is level 4's kernel behind the register-capped entry point.
          // It must not pick up level 6's reverse push: the launcher still runs
          // the (old,new) pass at this level, and the two together would
          // double-count candidates.
          if (pstore.n_cols() <=
              SALTATLAS_SOLANET_APU_NND_MIN_BLOCKS_DIM_THRESHOLD) {
            SALTATLAS_LAUNCH_NEIGHBOR_CHECK_CAPPED(
                4, SALTATLAS_SOLANET_APU_NND_MIN_BLOCKS_LOW_DIM);
          } else {
            SALTATLAS_LAUNCH_NEIGHBOR_CHECK_CAPPED(
                4, SALTATLAS_SOLANET_APU_NND_MIN_BLOCKS_HIGH_DIM);
          }
          break;
        case 6:
          if (pstore.n_cols() <=
              SALTATLAS_SOLANET_APU_NND_MIN_BLOCKS_DIM_THRESHOLD) {
            SALTATLAS_LAUNCH_NEIGHBOR_CHECK_CAPPED(
                6, SALTATLAS_SOLANET_APU_NND_MIN_BLOCKS_LOW_DIM);
          } else {
            SALTATLAS_LAUNCH_NEIGHBOR_CHECK_CAPPED(
                6, SALTATLAS_SOLANET_APU_NND_MIN_BLOCKS_HIGH_DIM);
          }
          break;
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        default:
          // Levels >= 8 change update_knng only, so the neighbour check runs
          // exactly as at level 6. Never fall through to a lower level here:
          // the launcher has already dropped the (old,new) pass, so a kernel
          // without the reverse push silently produces a worse graph.
          if (pstore.n_cols() <=
              SALTATLAS_SOLANET_APU_NND_MIN_BLOCKS_DIM_THRESHOLD) {
            SALTATLAS_LAUNCH_NEIGHBOR_CHECK_CAPPED(
                6, SALTATLAS_SOLANET_APU_NND_MIN_BLOCKS_LOW_DIM);
          } else {
            SALTATLAS_LAUNCH_NEIGHBOR_CHECK_CAPPED(
                6, SALTATLAS_SOLANET_APU_NND_MIN_BLOCKS_HIGH_DIM);
          }
          break;
      }
#undef SALTATLAS_LAUNCH_NEIGHBOR_CHECK_CAPPED
#undef SALTATLAS_LAUNCH_NEIGHBOR_CHECK
      SALTATLAS_HIP_CHECK(hipGetLastError());
      SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
      rec_time().stop();  // neighbor_checks

      launch_update_knng();
    };

    // Run neighor checks for (new, new), (new, old), and (old, new) neighbor
    // pairs.
    // (old, new) computes no distance that (new, old) has not already
    // computed: the metric is symmetric. It exists only because the kernel
    // writes results to nid1's list and never to nid2's. From level 6 the
    // kernel pushes both directions on the asymmetric pass, so the third launch
    // is redundant and its update_knng_with_candidates launch goes with it.
    neighbor_checker(new_ng, new_ng, /*symmetric_pass=*/true);
    neighbor_checker(new_ng, old_ng, /*symmetric_pass=*/false);
    if (nnd_opt_level < 6) {
      neighbor_checker(old_ng, new_ng, /*symmetric_pass=*/false);
    }

#ifdef SALTATLAS_SOLANET_APU_NND_PROFILE
    size_t total_candidates = 0;
    for (size_t i = 0; i < n_points; ++i) {
      total_candidates += std::min(candidate_counts[i], k);
    }
    std::cout << "Average #of candidates per point: "
              << (double(total_candidates) / n_points) << std::endl;
#endif

    // Copy the per-block update counters to the host explicitly. Reading
    // device memory directly from the host only works on unified-memory APUs.
    std::vector<size_t> h_n_updates_block(n_blocks);
    SALTATLAS_HIP_CHECK(hipMemcpy(h_n_updates_block.data(),
                                  n_updates_block.get(),
                                  n_blocks * sizeof(size_t),
                                  hipMemcpyDeviceToHost));
    size_t n_updates = 0;
    for (size_t i = 0; i < n_blocks; ++i) {
      n_updates += h_n_updates_block[i];
    }

    spdlog::trace("#of updates: {}", n_updates);
    if (n_updates <= double(delta) * size_t(k) * size_t(n_points)) {
      break;
    }
    ++super_step_no;
  }

  // Always erase MSB markers, including max-iteration termination.
  hipLaunchKernelGGL((clear_msbs<IDType>), grid_points, block, 0, nullptr,
                     knng_ids);
  SALTATLAS_HIP_CHECK(hipGetLastError());
  SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
}
}  // namespace

template <typename IDType, typename FEType, typename DistType>
std::pair<matrix<IDType>, matrix<DistType>> build_index(
    const matrix_view<FEType>& pstore, const std::string_view dist_func,
    const int k, const float rho, const float delta,
    const uint64_t seed = 0x12345678abcdefULL, const int max_iterations = 100) {
  if (dist_func == "l2") {
    return build_index<IDType, FEType, DistType, l2_distance_op>(
        pstore, k, rho, delta, seed, max_iterations);
  }

  if (dist_func == "cosine") {
    // hipVS does not support cosine distance.
    assert(false);
    return build_index<IDType, FEType, DistType, cosine_distance_op>(
        pstore, k, rho, delta, seed, max_iterations);
  }

  if (dist_func == "inner_product" || dist_func == "ip") {
    return build_index<IDType, FEType, DistType, inner_product_distance_op>(
        pstore, k, rho, delta, seed, max_iterations);
  }

  throw std::invalid_argument(
      "Only l2, cosine, and inner_product distances are supported.");
}

template <typename IDType, typename FEType, typename DistType, typename DistOp>
std::pair<matrix<IDType>, matrix<DistType>> build_index(
    const matrix_view<FEType>& pstore, const int k, const float rho,
    const float delta, const uint64_t seed = 0x12345678abcdefULL,
    const int max_iterations = 100) {
  static_assert(sizeof(IDType) == 4, "IDType must be 32-bit.");
  if (!pstore.data()) {
    throw std::invalid_argument("Point store is not initialized.");
  }

  const size_t n_points = pstore.n_rows();
  const int    dims     = pstore.n_cols();
  if (n_points == 0 || dims == 0 || k == 0) {
    return {};
  }
  if (n_points <= k) {
    throw std::invalid_argument("Number of points must be greater than k.");
  }
  if (max_iterations <= 0) {
    throw std::invalid_argument("max_iterations must be greater than 0.");
  }
  spdlog::trace("Build KNNG: n_points={}, dims={}, k={}, rho={}, delta={}",
                n_points, dims, k, rho, delta);

  int device_index = 0;
  SALTATLAS_HIP_CHECK(hipGetDevice(&device_index));
  hipDeviceProp_t device_prop{};
  SALTATLAS_HIP_CHECK(hipGetDeviceProperties(&device_prop, device_index));

  assert(device_prop.warpSize == k_warp_size);
  const dim3 block(k_nnd_block_size);
  if (block.x < k_warp_size) {
    throw std::runtime_error("Block size must be at least warp size.");
  }
  if (block.x % k_warp_size != 0) {
    throw std::runtime_error("Block size must be a multiple of warp size.");
  }
  // One point per thread.
  const size_t n_blocks = (n_points + block.x - 1) / block.x;
  if (n_blocks > device_prop.maxGridSize[0]) {
    throw std::runtime_error(
        "Grid size exceeds device limit. Reduce the number of points or "
        "increase "
        "the block size.");
  }
  // Grid for single point per thread.
  const dim3   grid_points(n_blocks);
  const size_t n_warps_per_block = block.x / k_warp_size;
  const size_t n_warp_blocks =
      (n_points + n_warps_per_block - 1) / n_warps_per_block;
  const size_t max_warp_blocks_by_tid =
      std::numeric_limits<uint32_t>::max() / static_cast<size_t>(block.x);
  const size_t n_warp_blocks_launch = std::min(
      n_warp_blocks, std::min(static_cast<size_t>(device_prop.maxGridSize[0]),
                              max_warp_blocks_by_tid));
  if (n_warp_blocks_launch == 0) {
    throw std::runtime_error("Neighbor check launch grid size became zero.");
  }
  spdlog::trace("Device: {}, block size: {}, grid size: {}", device_prop.name,
                block.x, grid_points.x);
  if (n_warp_blocks_launch < n_warp_blocks) {
    spdlog::trace(
        "Capped neighbor-check warp grid blocks from {} to {}. Kernels will "
        "iterate over points in warp-stride loops to stay within launch "
        "limits.",
        n_warp_blocks, n_warp_blocks_launch);
  }

  // Grid for single point per warp.
  const dim3 grid_warp_points(n_warp_blocks_launch);

  spdlog::trace("Initialize KNNG");
  rec_time().start("init_knng");
  spdlog::trace("Allocate KNNG ID");
  matrix<IDType> knng_ids(n_points, k);
  spdlog::trace("Allocate KNNG distance");
  matrix<DistType> knng_dists(n_points, k);
  hipLaunchKernelGGL((init_knng<IDType, FEType, DistType, DistOp>), grid_points,
                     block, 0, nullptr, pstore, k, seed, knng_ids.get_view(),
                     knng_dists.get_view());
  SALTATLAS_HIP_CHECK(hipGetLastError());
  SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
  rec_time().stop();  // init_knng

#ifdef NDEBUG
#ifdef SALTATLAS_SOLANET_APU_NND_EXTRA_CHECKS
  // Check if knng_dists are sorted
  for (size_t i = 0; i < n_points; ++i) {
    for (int j = 0; j < k - 1; ++j) {
      assert(knng_dists(i, j) <= knng_dists(i, j + 1));
    }
  }
#endif
#endif
  // matrix<bool> checked_flags(n_points, k);
  // std::memset(checked_flags.data(), 0, n_points * k * sizeof(bool));

  // NN-Descent main loop
  rec_time().start("nnd_main_loop");
  // #of new points to check for each point in each iteration
  const int p_new = static_cast<int>(k * rho);
  // #of reversed new neighbors for each point in each iteration
  const int p_r_new = static_cast<int>(k * rho);
  // #of old points to check for each point in each iteration
  const int p_old = static_cast<int>(k);
  // #of reversed old neighbors for each point in each iteration
  const int p_r_old = static_cast<int>(p_old * rho);

  spdlog::trace(
      "p_new: {}, p_r_new: {}, p_old: {}, p_r_old: {}. Total neighbors to "
      "check per iteration per point: {}",
      p_new, p_r_new, p_old, p_r_old, p_new + p_r_new + p_old + p_r_old);
  spdlog::trace("Allocate new neighbor list");
  matrix<IDType> new_ng(n_points, p_new + p_r_new);
  spdlog::trace("Allocate old neighbor list");
  matrix<IDType> old_ng(n_points, p_old + p_r_old);
  spdlog::trace("Allocate new counts array");
  auto new_counts = make_hip_array<int>(n_points);
  spdlog::trace("Allocate old counts array");
  auto old_counts = make_hip_array<int>(n_points);
  // Used to atomically add reverse neighbors
  auto new_counts_wk = make_hip_array<int>(n_points);
  auto old_counts_wk = make_hip_array<int>(n_points);

  build_index_main_loop<IDType, FEType, DistType, DistOp>(
      pstore, k, p_new, p_old, delta, max_iterations, device_prop, grid_points,
      grid_warp_points, block, n_blocks, knng_ids.get_view(),
      knng_dists.get_view(), old_ng.get_view(), old_counts, new_ng.get_view(),
      new_counts, old_counts_wk, new_counts_wk);
  rec_time().stop();  // nnd_main_loop

  return {std::move(knng_ids), std::move(knng_dists)};
}

}  // namespace saltatlas::solanet::apu_nn
