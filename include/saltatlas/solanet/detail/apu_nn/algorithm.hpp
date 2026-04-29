// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include <hip/hip_runtime.h>

#include "saltatlas/solanet/detail/apu_nn/utils.hpp"

namespace saltatlas::solanet::apu_nn {

template <typename KeyT>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE void swap_if_greater(KeyT& a,
                                                                  KeyT& b) {
  if (a > b) {
    std::swap(a, b);
  }
}

template <typename KeyT, typename ValT>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE void swap_if_greater(KeyT& a,
                                                                  ValT& av,
                                                                  KeyT& b,
                                                                  ValT& bv) {
  if (a > b) {
    std::swap(a, b);
    std::swap(av, bv);
  }
}

template <typename KeyT, typename ValT>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE void warp_compare_exchange(
    KeyT* keys, ValT* vals, int i, int j, bool ascending) {
  if constexpr (std::is_same<ValT, void>::value) {
    KeyT ki = keys[i];
    KeyT kj = keys[j];

    if (ascending) {
      swap_if_greater(ki, kj);
    } else {
      swap_if_greater(kj, ki);
    }

    keys[i] = ki;
    keys[j] = kj;
  } else {
    KeyT ki = keys[i];
    ValT vi = vals[i];
    KeyT kj = keys[j];
    ValT vj = vals[j];

    if (ascending) {
      swap_if_greater(ki, vi, kj, vj);
    } else {
      swap_if_greater(kj, vj, ki, vi);
    }

    keys[i] = ki;
    vals[i] = vi;
    keys[j] = kj;
    vals[j] = vj;
  }
}

template <typename KeyT, typename ValT, int M, int WARP = 64>
SALTATLAS_HD_DEVICE inline void warp_bitonic_sort(KeyT* keys, ValT* vals,
                                                  int n) {
  constexpr int NMAX = WARP * M;
  static_assert(M >= 1 && M <= 8,
                "warp_bitonic_sort supports 1..8 elements per lane.");
  static_assert((WARP & (WARP - 1)) == 0, "WARP must be a power of two");

  const int lane = threadIdx.x & (WARP - 1);

  if (n <= 1) {
    return;
  }
  if (n > NMAX) {
    n = NMAX;
  }
  assert((n & (n - 1)) == 0 && "warp_bitonic_sort requires power-of-two n");

  for (int k = 2; k <= n; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
#pragma unroll
      for (int t = 0; t < M; ++t) {
        const int i = lane * M + t;
        if (i >= n) {
          continue;
        }
        const int ixj = i ^ j;
        if (ixj > i && ixj < n) {
          const bool ascending = ((i & k) == 0);
          warp_compare_exchange(keys, vals, i, ixj, ascending);
        }
      }
      sync_warp();
    }
  }
}

template <typename KeyT, typename ValT, int M, int WARP = 64>
SALTATLAS_HD_DEVICE inline void warp_bitonic_sort(KeyT* keys,
                                                  ValT* vals = nullptr) {
  constexpr int N = WARP * M;
  static_assert((N & (N - 1)) == 0, "N must be power of two");
  warp_bitonic_sort<KeyT, ValT, M, WARP>(keys, vals, N);
}

/// \brief Simple Key-value pair sort.
// This is not a stable sort, and is only intended for small arrays.
template <typename KeyType, typename ValueType>
SALTATLAS_HD_HD inline void single_kv_sort(KeyType* const   keys,
                                           ValueType* const vals,
                                           const int        len) {
  for (int i = 0; i < len - 1; ++i) {
    for (int j = i + 1; j < len; ++j) {
      if (keys[j] < keys[i]) {
        std::swap(keys[i], keys[j]);
        if (vals != nullptr) {
          std::swap(vals[i], vals[j]);
        }
      }
    }
  }
}

template <typename KeyType, typename ValueType>
SALTATLAS_HD_HD inline void single_kv_sort_short(KeyType* const   keys,
                                                 ValueType* const vals,
                                                 const int        len) {
  if (len <= 1) {
    return;
  } else if (len == 2) {
    swap_if_greater(keys[0], vals[0], keys[1], vals[1]);
  } else if (len == 3) {
    swap_if_greater(keys[0], vals[0], keys[1], vals[1]);
    swap_if_greater(keys[1], vals[1], keys[2], vals[2]);
    swap_if_greater(keys[0], vals[0], keys[1], vals[1]);
  } else {
    single_kv_sort(keys, vals, len);
  }
}

// TODO: may not need this anymore.
// Single thread version
// Sort neighbors by distance
// If two neighbors have the same distance, sort by ID.
template <typename IDType, typename DistType>
SALTATLAS_HD_HD inline void sort_neighbors_single_thread(
    IDType* const nids, DistType* const dists, int k,
    const bool sort_by_distance = true) {
  for (int i = 0; i < k - 1; ++i) {
    for (int j = i + 1; j < k; ++j) {
      if (sort_by_distance) {
        if (dists[j] < dists[i] || (nearly_equal(dists[j], dists[i]) &&
                                    clear_msb(nids[j]) < clear_msb(nids[i]))) {
          std::swap(dists[i], dists[j]);
          std::swap(nids[i], nids[j]);
        }
      } else {
        if (clear_msb(nids[j]) < clear_msb(nids[i])) {
          std::swap(nids[i], nids[j]);
          std::swap(dists[i], dists[j]);
        }
      }
    }
  }
}

// Single thread version.
// Remove duplicate neighbor IDs, keeping the one with the smallest distance.
// Assumes neighbors are sorted by distance (and ID to break ties).
// Returns the number of unique neighbors.
template <typename IDType, typename DistType>
SALTATLAS_HD_HD inline int remove_duplicate_neighbors(IDType* const   nids,
                                                      DistType* const dists,
                                                      int             k) {
#if 0
  int unique_idx = 0;
  for (int i = 0; i < k; ++i) {
    const auto id   = clear_msb(nids[i]);
    bool       seen = false;
    for (int j = 0; j < unique_idx; ++j) {
      if (clear_msb(nids[j]) == id) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      nids[unique_idx]  = nids[i];
      dists[unique_idx] = dists[i];
      ++unique_idx;
    }
  }
  return unique_idx;
#else
  int unique_idx = 0;
  for (int i = 0; i < k; ++i) {
    if (i == 0 || clear_msb(nids[i]) != clear_msb(nids[i - 1])) {
      nids[unique_idx]  = nids[i];
      dists[unique_idx] = dists[i];
      ++unique_idx;
    }
  }
  return unique_idx;
#endif
}

namespace merge_and_keep_bes_detail {
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE int imin(int a, int b) {
  return a < b ? a : b;
}
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE int imax(int a, int b) {
  return a > b ? a : b;
}

// Merge-path partition: given diagonal d, find i such that i+j=d and
// A[i-1] <= B[j] and B[j-1] < A[i] (with bounds handled).
template <typename KeyT>
SALTATLAS_HD_DEVICE inline int merge_path_partition(const KeyT* A, int nA,
                                                    const KeyT* B, int nB,
                                                    int d) {
  int i_min = imax(0, d - nB);
  int i_max = imin(d, nA);

  while (i_min < i_max) {
    int i = (i_min + i_max) >> 1;
    int j = d - i;

    // A[i] and B[j-1]
    KeyT a_i   = (i < nA) ? A[i] : std::numeric_limits<KeyT>::max();
    KeyT b_jm1 = (j > 0) ? B[j - 1] : std::numeric_limits<KeyT>::min();

    if (b_jm1 > a_i)
      i_min = i + 1;
    else
      i_max = i;
  }
  return i_min;
}
}  // namespace merge_and_keep_bes_detail

/**
 * @brief Wave64 parallel merge that keeps the smallest nA elements in A.
 *
 * Given two sorted arrays A and B (ascending order), this function computes
 * the smallest nA elements of the merged sequence (A ∪ B) and stores them
 * back into A (sorted).
 *
 * This is a wave-cooperative implementation using the merge-path algorithm.
 * It partitions the first nA merge outputs across 64 lanes and lets each
 * lane merge a small contiguous segment independently.
 *
 * Key properties:
 *   - Does NOT require nA or nB to be power-of-two.
 *   - Works for any nA, nB ≤ 256.
 *   - Stable with respect to A (ties prefer A over B).
 *   - Time complexity: O(nA) total work.
 *   - Each lane performs ~nA / 64 serial merge steps.
 *
 * Memory requirements:
 *   - A, B must be sorted in ascending order.
 *   - tmpA must be a temporary buffer of size at least nA.
 *   - A_out may alias A (in-place overwrite is supported).
 *
 * Synchronization:
 *   - Requires block-level synchronization (__syncthreads())
 *     before copying tmpA back to A_out.
 *   - All 64 lanes must participate (wave-synchronous execution).
 *
 * Intended usage:
 *   - One wave64 handles one (A, B) merge.
 *   - Arrays typically reside in shared memory for best performance.
 *
 * Performance notes (MI300A):
 *   - Work-efficient (does not sort 512 elements).
 *   - Avoids unnecessary comparisons compared to bitonic re-sort.
 *   - Suitable for small arrays (≤256) in register/shared-memory scope.
 *
 * @param A      Pointer to sorted array A (size nA).
 * @param nA     Number of elements in A (output size).
 * @param B      Pointer to sorted array B (size nB).
 * @param nB     Number of elements in B.
 * @param tmpA   Temporary buffer of size nA.
 * @param A_out  Output buffer (can be same as A).
 */
template <typename KeyT, typename ValueT, int kWarpSize = 64>
SALTATLAS_HD_DEVICE inline void merge_and_keep_best(
    const KeyT* A_key, const ValueT* A_val, int nA, const KeyT* B_key,
    const ValueT* B_val, int nB, KeyT* tmp_key, ValueT* tmp_val,
    KeyT* A_key_out, ValueT* A_val_out) {
  const int lane = get_lane_id<kWarpSize>();

  if (nB > nA) {
    nB = nA;  // Only need to consider the first nA elements of B
  }

  // Partition first nA outputs across lanes
  int d0 = (nA * lane) / kWarpSize;
  int d1 = (nA * (lane + 1)) / kWarpSize;

  int a0 =
      merge_and_keep_bes_detail::merge_path_partition(A_key, nA, B_key, nB, d0);
  int a1 =
      merge_and_keep_bes_detail::merge_path_partition(A_key, nA, B_key, nB, d1);

  int b0 = d0 - a0;
  int b1 = d1 - a1;

  int i   = a0;
  int j   = b0;
  int out = d0;

  // Serial merge for this lane's segment
  while (out < d1) {
    KeyT ka = (i < a1) ? A_key[i] : std::numeric_limits<KeyT>::max();
    KeyT kb = (j < b1) ? B_key[j] : std::numeric_limits<KeyT>::max();

    bool takeA = false;
    // Use nearly_equal to handle floating-point distance ties
    if constexpr (std::is_floating_point<KeyT>::value) {
      takeA = (nearly_equal(ka, kb) || ka < kb);
    } else {
      takeA = (ka <= kb);  // stable: prefer A on equal
    }

    tmp_key[out] = takeA ? ka : kb;
    tmp_val[out] = takeA ? A_val[i] : B_val[j];

    i += int(takeA);
    j += int(!takeA);
    ++out;
  }
  sync_warp();

  // Copy back to A
  for (int idx = lane; idx < nA; idx += kWarpSize) {
    A_key_out[idx] = tmp_key[idx];
    A_val_out[idx] = tmp_val[idx];
  }
  sync_warp();
}

}  // namespace saltatlas::solanet::apu_nn
