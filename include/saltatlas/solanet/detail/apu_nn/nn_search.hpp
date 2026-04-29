// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "saltatlas/solanet/detail/apu_nn/algorithm.hpp"
#include "saltatlas/solanet/detail/apu_nn/distance.hpp"
#include "saltatlas/solanet/detail/apu_nn/matrix.hpp"
#include "saltatlas/solanet/detail/apu_nn/memory.hpp"
#include "saltatlas/solanet/detail/apu_nn/utils.hpp"

#ifndef SALTATLAS_SOLANET_APU_SEARCH_PREFETCH_SRC_FV
#define SALTATLAS_SOLANET_APU_SEARCH_PREFETCH_SRC_FV
#endif

#ifndef SALTATLAS_SOLANET_APU_NN_SEARCH_MAX_BUF_SIZE
#define SALTATLAS_SOLANET_APU_NN_SEARCH_MAX_BUF_SIZE 128
#endif

// #ifndef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
// #define SALTATLAS_SOLANET_APU_SEARCH_PROFILE
// #endif

// #define SALTATLAS_APU_NN_VERBOSE

#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <hip/hip_runtime.h>
#include <spdlog/spdlog.h>

#include "saltatlas/solanet/detail/apu_nn/hash_table.hpp"
#include "saltatlas/solanet/detail/apu_nn/nndescent.hpp"
#include "saltatlas/solanet/singleton_time_recorder.hpp"

namespace saltatlas::solanet::apu_nn {
namespace {
constexpr int k_cagra_block_size = 256;
// Maximun number of neighbors to search per point.
constexpr int k_max_search_buf_size =
    SALTATLAS_SOLANET_APU_NN_SEARCH_MAX_BUF_SIZE;
static constexpr int k_search_team_size = 8;
using visit_set_t                       = simple_set<uint32_t>;

template <typename IDType>
SALTATLAS_HD_DEVICE inline void select_initial_search_points(
    const size_t n_points, const int n_to_select, const uint64_t seed,
    IDType* out_ids, visit_set_t& visited) {
  rnd_state_type state;
  rnd_init(blockDim.x * blockIdx.x + threadIdx.x, seed, state);

  int n_selected = 0;
  while (n_selected < n_to_select) {
    const auto pid = static_cast<IDType>(rnd_next(state) % n_points);
    if (visited.add(pid)) {
      out_ids[n_selected] = pid;
      ++n_selected;
    }
  }
}

template <typename IDType, typename FEType, typename DistType>
SALTATLAS_HD_DEVICE inline void calculate_distances(
    const int lane_id, const matrix_view<FEType> pstore, const FEType* qfv,
    const int dims, const int n_neighbors, const IDType* nids,
    DistType* dists) {
  for (int i = lane_id; i < n_neighbors; i += warpSize) {
    const FEType* const nfv = pstore(nids[i]);
    dists[i]                = static_cast<DistType>(l2(qfv, nfv, dims));
  }
  sync_warp();
}

template <typename IDType, typename FEType, typename DistType>
SALTATLAS_HD_DEVICE inline void calculate_distances_team(
    const int lane_id, const matrix_view<FEType> pstore, const FEType* qfv,
    const int dims, const int n_neighbors, const IDType* nids,
    DistType* dists) {
  static constexpr auto k_invalid_id = std::numeric_limits<IDType>::max();
  const int             team_id =
      (threadIdx.x / k_search_team_size) % (warpSize / k_search_team_size);
  const int  n_teams     = (warpSize / k_search_team_size);
  const bool team_leader = (threadIdx.x % k_search_team_size) == 0;
  // Calculate distances in parallel within the warp
  for (int idx = team_id; idx < n_neighbors; idx += n_teams) {
    const IDType nid = nids[idx];
    if (nid == k_invalid_id) {
      if (team_leader) {
        dists[idx] = std::numeric_limits<DistType>::max();
      }
      continue;
    }
    const auto d =
        sql2_team<FEType, k_search_team_size>(qfv, pstore(nid), dims);
    if (team_leader) {
      dists[idx] = static_cast<DistType>(d);
    }
  }
  sync_warp();
}

template <typename IDType, typename DistType>
SALTATLAS_HD_DEVICE inline void merge_knng_lists(
    const int main_size, const int candidate_size, IDType* main_ids,
    DistType* main_dists, IDType* candidate_ids, DistType* candidate_dists) {
  static constexpr auto k_invalid_id = std::numeric_limits<IDType>::max();
  const int             lane_id      = threadIdx.x % warpSize;

  int cand_size = candidate_size;
  if (cand_size > main_size) {
    cand_size = main_size;
  }
  if (cand_size <= 0) {
    return;
  }

  if (candidate_dists[0] >= main_dists[main_size - 1]) {
    return;
  }

  for (int i = lane_id; i < (cand_size >> 1); i += warpSize) {
    const int j = cand_size - 1 - i;
    if (i < j) {
      std::swap(candidate_dists[i], candidate_dists[j]);
      std::swap(candidate_ids[i], candidate_ids[j]);
    }
  }
  sync_warp();

  const int total = main_size + cand_size;
  if (total <= 1) {
    return;
  }

  auto dist_at = [&](const int idx) -> DistType& {
    return (idx < main_size) ? main_dists[idx]
                             : candidate_dists[idx - main_size];
  };
  auto id_at = [&](const int idx) -> IDType& {
    return (idx < main_size) ? main_ids[idx] : candidate_ids[idx - main_size];
  };

  struct MergeTask {
    int lo;
    int len;
  };

  MergeTask stack[16];
  int       sp = 0;
  stack[sp++]  = {0, total};

  while (sp > 0) {
    const MergeTask task = stack[--sp];
    if (task.len <= 1) {
      continue;
    }

    const int m   = greatest_power_of_two_less_than(task.len);
    const int end = task.lo + task.len - m;
    for (int i = task.lo + lane_id; i < end; i += warpSize) {
      const int j   = i + m;
      DistType  di  = dist_at(i);
      DistType  dj  = dist_at(j);
      IDType    idi = id_at(i);
      IDType    idj = id_at(j);
      if (di > dj) {
        std::swap(di, dj);
        std::swap(idi, idj);
      }
      dist_at(i) = di;
      dist_at(j) = dj;
      id_at(i)   = idi;
      id_at(j)   = idj;
    }
    sync_warp();

    stack[sp++] = {task.lo, m};
    stack[sp++] = {task.lo + m, task.len - m};
  }
}

// Explore neighbors from the source points in frontier_ids
// NOTE: frontier_ids' MSB are set to 1 to indicate the point has been used as
// source point, and the original ID can be obtained by clear_msb(). Returns the
// number of neighbors added to next_ids.
template <typename IDType>
SALTATLAS_HD_DEVICE inline int explore_neighbors(
    const matrix_view<IDType> knng_ids, const int search_width,
    const int frontier_size, IDType* const frontier_ids, IDType* const next_ids,
    visit_set_t& visited_set) {
  static constexpr auto k_invalid_id = std::numeric_limits<IDType>::max();
  // Only one thred in the warp calls this function
  assert((threadIdx.x % warpSize) == 0);
  assert(search_width <= frontier_size);

  int source_count = 0;
  int next_count   = 0;
  for (int i = 0; i < frontier_size && source_count < search_width; ++i) {
    const IDType sid = frontier_ids[i];
    if (get_msb<IDType>(sid)) {
      continue;
    }
    // Mark as used
    frontier_ids[i] = set_msb<IDType>(sid);
    ++source_count;

    // Visit sid's neighbors
    for (int j = 0; j < knng_ids.n_cols(); ++j) {
      const IDType nid = knng_ids(sid, j);
      if (visited_set.add(nid)) {
        next_ids[next_count] = nid;
        ++next_count;
      }
    }
  }

  return next_count;
}

// NOTE: This function assumes that 'init_frontier_size' is small enough to be
// sorted by a single thread.
template <typename IDType, typename FEType, typename DistType>
__global__ void search_kernel(
    const matrix_view<FEType> pstore, const matrix_view<IDType> knng_ids,
    const matrix_view<FEType> queries, const int k, const int search_width,
    const int init_frontier_size, const int frontier_size, const int next_cap,
    const int buf_cap, const int visited_set_cap,
    const int visit_tbl_reset_interval, const uint64_t rnd_seed,
    matrix_view<IDType> out_ids, matrix_view<DistType> out_dists) {
  const int n_local_warps = blockDim.x / warpSize;
  if (n_local_warps <= 0) {
    return;
  }

  const size_t n_g_threads =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
      static_cast<size_t>(threadIdx.x);
  const size_t g_warp_id = n_g_threads / warpSize;
  const int    l_warp_id = threadIdx.x / warpSize;
  const int    lane_id   = threadIdx.x % warpSize;

  if (g_warp_id >= queries.n_rows()) {
    return;
  }

  // Allocate shared memory
  extern __shared__ unsigned char shared_storage[];
  unsigned char*                  shared_ptr = shared_storage;
  IDType*                         buf_ids =
      get_shared_mem<IDType>(shared_ptr, buf_cap, n_local_warps, l_warp_id);
  DistType* buf_dists =
      get_shared_mem<DistType>(shared_ptr, buf_cap, n_local_warps, l_warp_id);
  IDType*   frontier_ids   = buf_ids;
  DistType* frontier_dists = buf_dists;
  IDType*   next_ids       = buf_ids + frontier_size;
  DistType* next_dists     = buf_dists + frontier_size;

  IDType*     visit_buf = get_shared_mem<IDType>(shared_ptr, visited_set_cap,
                                                 n_local_warps, l_warp_id);
  visit_set_t visited(visited_set_cap, visit_buf);
  if (lane_id == 0) {
    visited.clear();
  }

#ifdef SALTATLAS_SOLANET_APU_SEARCH_PREFETCH_SRC_FV
  FEType* qfv = get_shared_mem<FEType>(shared_ptr, pstore.n_cols(),
                                       n_local_warps, l_warp_id);
  {
    const FEType* const sfvec_global = queries(g_warp_id);
    for (int i = lane_id; i < pstore.n_cols(); i += warpSize) {
      qfv[i] = sfvec_global[i];
    }
  }
  sync_warp();
#else
  FEType* qfv = queries(g_warp_id);
#endif

  IDType* frontier_ids_work_buf = get_shared_mem<IDType>(
      shared_ptr, frontier_size, n_local_warps, l_warp_id);
  DistType* frontier_dists_work_buf = get_shared_mem<DistType>(
      shared_ptr, frontier_size, n_local_warps, l_warp_id);

#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
  const auto st = clock64();
#endif
  if (lane_id == 0) {
    select_initial_search_points<IDType>(pstore.n_rows(), init_frontier_size,
                                         rnd_seed + g_warp_id, frontier_ids,
                                         visited);
  }
  // Fill the rest of frontier with invalid IDs and max distances.
  for (int i = init_frontier_size + lane_id; i < frontier_size; i += warpSize) {
    frontier_ids[i]   = static_cast<IDType>(-1);
    frontier_dists[i] = std::numeric_limits<DistType>::max();
  }
  sync_warp();

  // First iteration: calculate distances for initial frontier points
  calculate_distances<IDType, FEType, DistType>(
      lane_id, pstore, qfv, pstore.n_cols(), init_frontier_size, frontier_ids,
      frontier_dists);
  if (lane_id == 0) {
    // sort frontier neighbors by distance
    // We assubme init_frontier_size is small and use bubble sort for
    // simplicity.
    single_kv_sort_short(frontier_dists, frontier_ids, init_frontier_size);
  }
  sync_warp();

#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
  if (g_warp_id == 0 && lane_id == 0) {
    printf("Fill search frontier took: %lld\n", (clock64() - st) / 1000);
  }
#endif

  int next_size = 0;
#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
  const auto main_kernel_st = clock64();
#endif
  for (int iter = 0;; ++iter) {
#ifdef SALTATLAS_APU_NN_VERBOSE
    if (g_warp_id == 0 && lane_id == 0) {
      printf("Iteration %d start\n", iter);
    }
#endif

    // Reset visit table periodically
    if (iter % visit_tbl_reset_interval == 0 && iter > 0) {
      if (lane_id == 0) {
#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
        const auto st = clock64();
#endif
        visited.clear();
        for (int i = 0; i < frontier_size; ++i) {
          visited.add(clear_msb(frontier_ids[i]));
        }
#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
        if (g_warp_id == 0 && lane_id == 0) {
          printf("  reset hash table took: %lld\n", (clock64() - st) / 1000);
        }
#endif
      }
    }

    // Explore neighbors from the frontier points and put them in next ids.
    // Use 'search_width' closest points as source points and visit their
    // neighbors. Selected source points are marked to indicate they have beeen
    // used --- their MSBs are set to 1.
    if (lane_id == 0) {
#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
      const auto st = clock64();
#endif
      next_size =
          explore_neighbors<IDType>(knng_ids, search_width, frontier_size,
                                    frontier_ids, next_ids, visited);
      assert(next_size <= next_cap);
#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
      if (g_warp_id == 0 && lane_id == 0) {
        printf("  explore_neighbors took: %lld\n", (clock64() - st) / 1000);
      }
#endif
#ifdef SALTATLAS_APU_NN_VERBOSE
      if (g_warp_id == 0 && lane_id == 0) {
        printf("  next_size: %d\n", next_size);
      }
#endif
    }
    next_size = __shfl(next_size, 0, warpSize);
    // Terminate the search if no new points were found
    if (next_size == 0) {
      break;
    }

    // -----
    // Calculate distances for points in the next array.
    // -----
    {
#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
      const auto st = clock64();
#endif
      calculate_distances(lane_id, pstore, qfv, pstore.n_cols(), next_size,
                          next_ids, next_dists);
#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
      if (g_warp_id == 0 && lane_id == 0) {
        printf("  calculate_distances took: %lld\n", (clock64() - st) / 1000);
      }
#endif
    }

// -----
// Sort frontier and next as a one array.
// -----
#if 0
    {
#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
      const auto st = clock64();
#endif

      // Fill the rest of next with invalid IDs and max distances
      for (int i = next_size + lane_id; i < next_cap; i += warpSize) {
        next_ids[i]   = static_cast<IDType>(-1);
        next_dists[i] = std::numeric_limits<DistType>::max();
      }
      sync_warp();

      // First, sort only the next cadidates
      warp_bitonic_sort<DistType, IDType, (k_max_search_buf_size / k_warp_size)>(
          next_dists, next_ids, next_cap);
#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
      if (g_warp_id == 0 && lane_id == 0) {
        printf("  sort_neighbors took: %lld\n", (clock64() - st) / 1000);
      }
#endif
    }

    {
#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
      const auto st = clock64();
      if (g_warp_id == 0 && lane_id == 0) {
        printf("  start merging frontier and next %d %d\n", frontier_size,
               next_size);
      }
#endif
      // Then merge the sorted next candidates with the frontier
      // into a sorted order. We only care about the top 'frontier_size' points
      merge_and_keep_best<DistType, IDType>(
          frontier_dists, frontier_ids, frontier_size, next_dists, next_ids,
          next_size, frontier_dists_work_buf, frontier_ids_work_buf,
          frontier_dists, frontier_ids);
      // merge_knng_lists(frontier_ids.size(), next_size, frontier_ids.data(),
      //                  frontier_dists.data(), next_ids.data(),
      //                  next_dists.data());
#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
      if (g_warp_id == 0 && lane_id == 0) {
        printf("  merge_knng_lists took: %lld\n", (clock64() - st) / 1000);
      }
#endif
    }
#else
    {
#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
      const auto st = clock64();
#endif
      // DB
      // if (buf_cap > k_max_search_buf_size) {
      //   // This should not happen as we check the parameters at the beginning
      //   of
      //   // run_search(), but we add this assert here just in case.
      //   if (lane_id == 0) {
      //     printf(
      //         "Error: buf_cap (%d) exceeds k_max_search_buf_size (%d). This "
      //         "may be "
      //         "caused by "
      //         "invalid parameters. Please check your parameters.\n",
      //         buf_cap, k_max_search_buf_size);
      //   }
      //   return;
      // }

      for (int i = frontier_size + next_size + lane_id; i < buf_cap;
           i += warpSize) {
        buf_ids[i]   = static_cast<IDType>(-1);
        buf_dists[i] = std::numeric_limits<DistType>::max();
      }
      sync_warp();
      warp_bitonic_sort<DistType, IDType,
                        (k_max_search_buf_size / k_warp_size)>(
          buf_dists, buf_ids, buf_cap);
#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
      if (g_warp_id == 0 && lane_id == 0) {
        printf("  merge_knng_lists took: %lld\n", (clock64() - st) / 1000);
      }
#endif
    }
#endif
  }  // end of iterations
#ifdef SALTATLAS_SOLANET_APU_SEARCH_PROFILE
  if (g_warp_id == 0 && lane_id == 0) {
    printf("Main loop took: %lld\n", (clock64() - main_kernel_st) / 1000);
  }
#endif

  // Write back results to global memory
  for (int i = lane_id; i < k; i += warpSize) {
    out_ids(g_warp_id, i)   = clear_msb(frontier_ids[i]);
    out_dists(g_warp_id, i) = frontier_dists[i];
  }
}
}  // namespace

/// \brief Run APU-based approximate nearest neighbor search
/// \tparam IDType          Data type for point IDs
/// \tparam FEType Data type for feature elements
/// \tparam DistType    Data type for distances
/// \param pstore          Point store matrix view
/// \param knng_ids        Initial k-NN graph IDs matrix view
/// \param queries         Query points matrix view
/// \param k               Number of neighbors to search for
/// \param epsilon         Approximation factor. Used to determine the size of
/// the search frontier.
/// \param search_width           Number of points to visit per search iteration
/// \param init_frontier_size     Number of random points to initialize the
/// search frontier with
/// \param visit_info_reset_interval Number of iterations after which the
/// visited hash table is reset.
/// \param rnd_seed        Random seed for search initialization
/// \param verbose         If true, print verbose logs
/// \param out_ids         Output matrix view for neighbor IDs
/// \param out_dists       Output matrix view for neighbor distances
template <typename IDType, typename FEType, typename DistType>
void run_search(const matrix_view<FEType> pstore,
                const matrix_view<IDType> knng_ids,
                const matrix_view<FEType> queries, const int k,
                const float epsilon, const int search_width,
                const int init_frontier_size,
                const int visit_info_reset_interval, const uint64_t rnd_seed,
                const bool verbose, matrix_view<IDType> out_ids,
                matrix_view<DistType> out_dists) {
  if (pstore.n_rows() == 0 || queries.n_rows() == 0 || k <= 0) {
    throw std::invalid_argument("run_search: invalid arguments.");
  }
  if (pstore.n_cols() != queries.n_cols()) {
    throw std::invalid_argument("run_search: dimension mismatch.");
  }
  if (pstore.n_rows() < static_cast<size_t>(k)) {
    throw std::invalid_argument("run_search: there are fewer points than k.");
  }
  if (k > k_max_search_buf_size) {
    throw std::invalid_argument(
        "run_search: k is too large. Maximum supported k is " +
        std::to_string(k_max_search_buf_size) + ".");
  }
  if (search_width <= 0) {
    throw std::invalid_argument("run_search: search_width must be positive.");
  }
  if (init_frontier_size <= 0) {
    throw std::invalid_argument(
        "run_search: init_frontier_size must be positive.");
  }
  if (search_width > init_frontier_size) {
    throw std::invalid_argument(
        "run_search: search_width must be <= init_frontier_size.");
  }

  const int degree = knng_ids.n_cols();
  if (degree <= 0) {
    throw std::invalid_argument(
        "run_search: knng_ids must have at least one neighbor per point.");
  }
  assert(epsilon >= 0.0f && "epsilon must be non-negative");
  const int frontier_size = k * (1.0f + epsilon);
  const int next_size     = degree * search_width;
  const int next_cap      = next_power_of_two(next_size);
  if (verbose) {
    std::cout << "frontier_size: " << frontier_size
              << ", next_cap: " << next_cap << std::endl;
  }
  const int    buf_cap = next_power_of_two(frontier_size + next_cap);
  const double visited_set_max_load_factor = 0.5;
  const int    visited_set_cap =
      frontier_size + next_cap * visit_info_reset_interval *
                          (1.0 / visited_set_max_load_factor);
  if (frontier_size > k_max_search_buf_size) {
    throw std::invalid_argument(
        "run_search: frontier_size exceeds k_max_search_buf_size; reduce "
        "epsilon or "
        "k.");
  }
  if (next_cap > k_max_search_buf_size) {
    throw std::invalid_argument(
        "run_search: next_cap exceeds k_max_search_buf_size; reduce "
        "search_width or "
        "degree.");
  }

  // Determine shared memory size
  hipDeviceProp_t device_prop;
  int             device_index = 0;
  SALTATLAS_HIP_CHECK(hipGetDevice(&device_index));
  SALTATLAS_HIP_CHECK(hipGetDeviceProperties(&device_prop, device_index));
  const int warp_size = device_prop.warpSize;
  if (warp_size <= 0) {
    throw std::runtime_error("run_search: invalid HIP warp size.");
  }

  const dim3 block(k_cagra_block_size);
  if (block.x < static_cast<unsigned int>(warp_size) ||
      block.x % static_cast<unsigned int>(warp_size) != 0) {
    throw std::runtime_error(
        "run_search: block size must be a multiple of warp size.");
  }
  const size_t n_local_warps    = block.x / warp_size;
  const size_t shared_mem_bytes = [&]() {
    size_t offset = 0;

    // Buffer for IDs
    offset = align_up(offset, alignof(IDType));
    offset += n_local_warps * static_cast<size_t>(buf_cap) * sizeof(IDType);

    // Buffer for distances
    offset = align_up(offset, alignof(DistType));
    offset += n_local_warps * static_cast<size_t>(buf_cap) * sizeof(DistType);

    // Visited set
    offset = align_up(offset, alignof(IDType));
    offset +=
        n_local_warps * static_cast<size_t>(visited_set_cap) * sizeof(IDType);

    // Optional: buffer for source feature vector
#ifdef SALTATLAS_SOLANET_APU_SEARCH_PREFETCH_SRC_FV
    offset = align_up(offset, alignof(FEType));
    offset += n_local_warps * pstore.n_cols() * sizeof(FEType);
#endif

    // Buffer used for merging frontier and next candidates.
    offset = align_up(offset, alignof(IDType));
    offset +=
        n_local_warps * static_cast<size_t>(frontier_size) * sizeof(IDType);
    offset = align_up(offset, alignof(DistType));
    offset +=
        n_local_warps * static_cast<size_t>(frontier_size) * sizeof(DistType);

    return offset;
  }();
  if (shared_mem_bytes > device_prop.sharedMemPerBlock) {
    throw std::runtime_error(
        "run_search: required shared memory exceeds device limit.");
  }
  if (verbose) {
    std::cout << "APU run_search shared memory (KiB): "
              << shared_mem_bytes / 1024.0 << std::endl;
  }

  const size_t n_points = queries.n_rows();
  // Each warp processes one query point
  const size_t n_warps = (n_points + n_local_warps - 1) / n_local_warps;
  const dim3   grid(n_warps);
  hipLaunchKernelGGL((search_kernel<IDType, FEType, DistType>), grid, block,
                     shared_mem_bytes, nullptr, pstore, knng_ids, queries, k,
                     search_width, init_frontier_size, frontier_size, next_cap,
                     buf_cap, visited_set_cap, visit_info_reset_interval,
                     rnd_seed, out_ids, out_dists);
  SALTATLAS_HIP_CHECK(hipGetLastError());
  SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
}
}  // namespace saltatlas::solanet::apu_nn
