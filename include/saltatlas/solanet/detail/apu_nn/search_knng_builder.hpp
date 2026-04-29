// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>
#include <spdlog/spdlog.h>
#include <sys/types.h>
#include <boost/unordered/unordered_flat_map.hpp>

#include "saltatlas/dnnd/detail/knn_heap.hpp"
#include "saltatlas/dnnd/detail/utilities/omp.hpp"
#include "saltatlas/solanet/detail/apu_nn/algorithm.hpp"
#include "saltatlas/solanet/detail/apu_nn/csr.hpp"
#include "saltatlas/solanet/detail/apu_nn/graph_reverser.hpp"
#include "saltatlas/solanet/detail/apu_nn/hash_table.hpp"
#include "saltatlas/solanet/detail/apu_nn/matrix.hpp"
#include "saltatlas/solanet/detail/apu_nn/memory.hpp"
#include "saltatlas/solanet/detail/apu_nn/utils.hpp"
#include "saltatlas/solanet/detail/nn_index_view.hpp"

#ifndef SALTATLAS_SOLANET_APU_NN_QUERY_GRAPH_MAX_K
#define SALTATLAS_SOLANET_APU_NN_QUERY_GRAPH_MAX_K 128
#endif

namespace saltatlas::solanet::apu_nn {
enum class query_graph_distance_mode : uint8_t {
  actual_distance = 0,
  knng_position   = 1
};

namespace detail {
constexpr int k_query_graph_max_k = SALTATLAS_SOLANET_APU_NN_QUERY_GRAPH_MAX_K;
constexpr int k_query_block_size  = 256;

template <typename id_type, typename dist_type, bool UseKnngPositionAsDistance>
SALTATLAS_HD_GLOBAL void build_ranked_knng_kernel(
    const matrix_view<const id_type>   in_knng_ids,
    const matrix_view<const dist_type> in_knng_dists,
    matrix_view<id_type> ranked_knng, matrix_view<dist_type> ranked_knng_ranks,
    const size_t map_capacity, id_type* map_keys, int* map_vals) {
  const int tid    = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int stride = static_cast<int>(gridDim.x * blockDim.x);
  const int n_rows = static_cast<int>(in_knng_ids.n_rows());
  const int k      = static_cast<int>(in_knng_ids.n_cols());
  if (k == 0) {
    return;
  }

  id_type ids_local[k_query_graph_max_k];
  int     counts_local[k_query_graph_max_k];

  for (id_type sid = tid; sid < n_rows; sid += stride) {
    simple_map<id_type, int> pos_map(map_capacity,
                                     map_keys + sid * map_capacity,
                                     map_vals + sid * map_capacity);
    pos_map.clear();

    for (int i = 0; i < k; ++i) {
      const auto nid = in_knng_ids(sid, i);
      pos_map.insert(nid, i);
      ids_local[i]    = nid;
      counts_local[i] = 0;
    }

    for (int i = 0; i < k; ++i) {
      const auto tid = in_knng_ids(sid, i);
      if constexpr (UseKnngPositionAsDistance) {
        for (int j = 0; j < k; ++j) {
          const auto zid     = in_knng_ids(tid, j);
          const int  pos_zid = pos_map.get(zid, -1);
          if (pos_zid >= 0 && pos_zid > i && pos_zid > j) {
            ++counts_local[pos_zid];
          }
        }
      } else {
        const auto dist_st = in_knng_dists(sid, i);
        for (int j = 0; j < k; ++j) {
          const auto zid     = in_knng_ids(tid, j);
          const int  pos_zid = pos_map.get(zid, -1);
          if (pos_zid < 0) {
            continue;
          }
          const auto dist_tz = in_knng_dists(tid, j);
          const auto dist_sz = in_knng_dists(sid, pos_zid);
          if (dist_sz > dist_st && dist_sz > dist_tz) {
            ++counts_local[pos_zid];
          }
        }
      }
    }

    for (int i = 0; i < k; ++i) {
      int min_pos = i;
      int min_val = counts_local[i];
      for (int j = i + 1; j < k; ++j) {
        if (counts_local[j] < min_val) {
          min_val = counts_local[j];
          min_pos = j;
        }
      }
      if (min_pos != i) {
        const auto tmp_id     = ids_local[i];
        const auto tmp_count  = counts_local[i];
        ids_local[i]          = ids_local[min_pos];
        counts_local[i]       = counts_local[min_pos];
        ids_local[min_pos]    = tmp_id;
        counts_local[min_pos] = tmp_count;
      }
    }

    for (int i = 0; i < k; ++i) {
      ranked_knng(sid, i)       = ids_local[i];
      ranked_knng_ranks(sid, i) = static_cast<dist_type>(i);
    }
  }
}

template <typename id_type, typename dist_type>
SALTATLAS_HD_GLOBAL void merge_ranked_knng_kernel(
    const matrix_view<const id_type> ranked_knng, const id_type* r_offsets,
    const id_type* r_ids, const dist_type* r_ranks,
    matrix_view<id_type> out_query_nids, const size_t set_capacity,
    id_type* set_keys) {
  const int tid    = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int stride = static_cast<int>(gridDim.x * blockDim.x);
  const int n_rows = static_cast<int>(ranked_knng.n_rows());
  const int in_k   = static_cast<int>(ranked_knng.n_cols());
  const int out_k  = static_cast<int>(out_query_nids.n_cols());
  if (in_k == 0 || out_k == 0) {
    return;
  }
  const int half = out_k / 2;

  id_type   best_ids[k_query_graph_max_k];
  dist_type best_ranks[k_query_graph_max_k];
  id_type   out_ids[k_query_graph_max_k];
  dist_type out_ranks[k_query_graph_max_k];

  for (int sid = tid; sid < n_rows; sid += stride) {
    simple_set<id_type> used(set_capacity, set_keys + sid * set_capacity);
    used.clear();

    // Seed output with half of the output budget from ranked KNNG.
    int out_pos = 0;
    for (int i = 0; i < half && i < in_k && out_pos < out_k; ++i) {
      const auto nid = ranked_knng(sid, i);
      if (used.add(nid)) {
        out_ids[out_pos]   = nid;
        out_ranks[out_pos] = static_cast<dist_type>(i);
        ++out_pos;
      }
    }

    // Collect the best half of reverse neighbors by rank using insertion.
    int          best_size = 0;
    const size_t begin     = static_cast<size_t>(r_offsets[sid]);
    const size_t end       = static_cast<size_t>(r_offsets[sid + 1]);
    for (size_t idx = begin; idx < end; ++idx) {
      const auto id   = r_ids[idx];
      const auto rank = r_ranks[idx];
      int        pos  = -1;
      if (best_size < half) {
        pos = best_size++;
      } else if (half > 0 && rank < best_ranks[half - 1]) {
        pos = half - 1;
      }
      if (pos >= 0) {
        while (pos > 0 && rank < best_ranks[pos - 1]) {
          best_ranks[pos] = best_ranks[pos - 1];
          best_ids[pos]   = best_ids[pos - 1];
          --pos;
        }
        best_ranks[pos] = rank;
        best_ids[pos]   = id;
      }
    }

    // Merge reverse-neighbor candidates, keeping uniqueness.
    for (int i = 0; i < best_size && out_pos < out_k; ++i) {
      const auto nid = best_ids[i];
      if (used.add(nid)) {
        out_ids[out_pos]   = nid;
        out_ranks[out_pos] = best_ranks[i];
        ++out_pos;
      }
    }

    // Fill remaining slots from ranked KNNG.
    for (int i = half; out_pos < out_k && i < in_k; ++i) {
      const auto nid = ranked_knng(sid, i);
      if (used.add(nid)) {
        out_ids[out_pos]   = nid;
        out_ranks[out_pos] = static_cast<dist_type>(i);
        ++out_pos;
      }
    }

    // Sort selected neighbors by rank (ascending) for deterministic truncation.
    for (int i = 1; i < out_pos; ++i) {
      const auto key_id   = out_ids[i];
      const auto key_rank = out_ranks[i];
      int        j        = i - 1;
      while (j >= 0 && (out_ranks[j] > key_rank ||
                        (out_ranks[j] == key_rank && out_ids[j] > key_id))) {
        out_ids[j + 1]   = out_ids[j];
        out_ranks[j + 1] = out_ranks[j];
        --j;
      }
      out_ids[j + 1]   = key_id;
      out_ranks[j + 1] = key_rank;
    }

    // Write out up to out_k neighbors after sorting by rank.
    const int write_k = (out_pos < out_k) ? out_pos : out_k;
    for (int i = 0; i < write_k; ++i) {
      out_query_nids(sid, i) = out_ids[i];
    }
  }
}
}  // namespace detail

// Wrapper that builds a bounded reverse CSR graph from dense KNNG arrays.
template <typename id_type, typename dist_type>
inline csr_graph<id_type, dist_type> make_reversed_graph(
    const matrix_view<id_type>&   in_knng_ids,
    const matrix_view<dist_type>& in_knng_values,
    const size_t                  max_edges_per_vertex) {
  return make_reversed_graph_apu<id_type, dist_type>(
      in_knng_ids, in_knng_values, max_edges_per_vertex);
}

// Wrapper that builds reverse CSR with default cap (2x forward degree).
template <typename id_type, typename dist_type>
inline csr_graph<id_type, dist_type> make_reversed_graph(
    const matrix_view<id_type>&   in_knng_ids,
    const matrix_view<dist_type>& in_knng_values) {
  return make_reversed_graph_apu<id_type, dist_type>(in_knng_ids,
                                                     in_knng_values);
}

template <typename id_type, typename dist_type>
inline void make_optimized_query_graph_apu(
    const matrix_view<id_type>&     in_knng_ids,
    const matrix_view<dist_type>&   in_knng_dists,
    matrix_view<id_type>            out_query_nids,
    const query_graph_distance_mode distance_mode =
        query_graph_distance_mode::actual_distance) {
  const int n_pts = static_cast<int>(in_knng_ids.n_rows());
  const int in_k  = static_cast<int>(in_knng_ids.n_cols());
  const int out_k = static_cast<int>(out_query_nids.n_cols());
  if (n_pts == 0 || in_k == 0 || out_k == 0) {
    return;
  }
  assert(out_query_nids.n_rows() == static_cast<size_t>(n_pts));
  if (out_k > in_k) {
    throw std::invalid_argument(
        "make_optimized_query_graph_apu: output k exceeds input k.");
  }
  if (in_k > detail::k_query_graph_max_k) {
    throw std::invalid_argument(
        "make_optimized_query_graph_apu: k exceeds APU query graph limit.");
  }

  spdlog::trace("Count detours");
  rec_time().start("Count-detours");

  spdlog::trace("Allocate ranked KNNG");
  matrix<id_type>   ranked_knng(n_pts, in_k);
  matrix<dist_type> ranked_knng_ranks(n_pts, in_k);

  // Allocate extra space to keep the load factor of the map low (50%).
  spdlog::trace("Allocate map for counting detours");
  const int map_capacity = in_k * 2 + 1;
  auto      map_keys =
      make_hip_array<id_type>(static_cast<size_t>(n_pts) * map_capacity);
  auto map_vals =
      make_hip_array<int>(static_cast<size_t>(n_pts) * map_capacity);

  const auto grid = detail::make_grid_1d(n_pts, detail::k_query_block_size);
  if (distance_mode == query_graph_distance_mode::knng_position) {
    hipLaunchKernelGGL(
        (detail::build_ranked_knng_kernel<id_type, dist_type, true>), grid,
        dim3(detail::k_query_block_size), 0, nullptr,
        matrix_view<const id_type>(in_knng_ids.data(), in_knng_ids.n_rows(),
                                   in_knng_ids.n_cols()),
        matrix_view<const dist_type>(in_knng_dists.data(),
                                     in_knng_dists.n_rows(),
                                     in_knng_dists.n_cols()),
        ranked_knng.get_view(), ranked_knng_ranks.get_view(), map_capacity,
        map_keys.get(), map_vals.get());
  } else {
    hipLaunchKernelGGL(
        (detail::build_ranked_knng_kernel<id_type, dist_type, false>), grid,
        dim3(detail::k_query_block_size), 0, nullptr,
        matrix_view<const id_type>(in_knng_ids.data(), in_knng_ids.n_rows(),
                                   in_knng_ids.n_cols()),
        matrix_view<const dist_type>(in_knng_dists.data(),
                                     in_knng_dists.n_rows(),
                                     in_knng_dists.n_cols()),
        ranked_knng.get_view(), ranked_knng_ranks.get_view(), map_capacity,
        map_keys.get(), map_vals.get());
  }
  SALTATLAS_HIP_CHECK(hipGetLastError());
  rec_time().stop();  // Count-detours

  // Release map memory as it's no longer needed after building the ranked KNNG.
  map_keys.reset();
  map_vals.reset();

  spdlog::trace("Make reverse ranked knng");
  rec_time().start("Make-reverse-ranked-knng");
  const int reverse_degree_cap =
      in_k * 4;  // Cap for reverse degree to control memory usage
  auto r_ranked_knng = make_reversed_graph<id_type, dist_type>(
      ranked_knng.get_view(), ranked_knng_ranks.get_view(), reverse_degree_cap);
  rec_time().stop();  // Make-reverse-ranked-knng

  spdlog::trace("Merge ranked knng and reverse ranked knng");
  rec_time().start("Merge-ranked-knngs");

  // TODO: try to reduce memory
  // in_k * 2 + 1 is too large
  const int set_cap = in_k * 2 + 1;
  auto set_keys = make_hip_array<id_type>(static_cast<size_t>(n_pts) * set_cap);

  hipLaunchKernelGGL(
      (detail::merge_ranked_knng_kernel<id_type, dist_type>), grid,
      dim3(detail::k_query_block_size), 0, nullptr,
      matrix_view<const id_type>(ranked_knng.data(), ranked_knng.n_rows(),
                                 ranked_knng.n_cols()),
      r_ranked_knng.offsets.get(), r_ranked_knng.ids.get(),
      r_ranked_knng.distances.get(), out_query_nids, set_cap, set_keys.get());
  SALTATLAS_HIP_CHECK(hipGetLastError());
  rec_time().stop();  // Merge-ranked-knngs
}

template <typename id_type, typename dist_type>
inline void make_optimized_query_graph(
    const matrix_view<id_type>&   in_knng_ids,
    const matrix_view<dist_type>& in_knng_dists,
    matrix_view<id_type>          out_query_nids) {
  using knn_heap_t          = dndetail::unique_knn_heap<id_type, int>;
  using knn_heap_adj_list_t = std::vector<knn_heap_t>;
  const int n_points        = in_knng_ids.n_rows();
  // Assumes that out_query_nids is already allocated
  assert(out_query_nids.n_rows() == n_points);
  assert(out_query_nids.n_cols() == in_knng_ids.n_cols());

#ifndef NDEBUG
  // Check knng does not contain duplicates neighbors
    OMP_DIRECTIVE(parallel for)
    for (int i = 0; i < n_points; ++i) {
      std::unordered_set<id_type> neighbor_set;
      for (int j = 0; j < in_knng_ids.n_cols(); ++j) {
        const auto neighbor_id = in_knng_ids(i, j);
        assert(neighbor_id != static_cast<id_type>(-1));
        if (neighbor_set.count(neighbor_id) > 0) {
          std::cerr << __FILE__ << " " << __LINE__
                    << ": Error: duplicate neighbor found in knng_ids at row "
                    << i << ": " << neighbor_id << std::endl;
        }
        assert(neighbor_set.count(neighbor_id) == 0);
        neighbor_set.insert(neighbor_id);
      }
    }
#endif

    spdlog::trace("Count detours");
    rec_time().start("Count-detours");
    matrix<int> detour_counts_table(n_points, in_knng_ids.n_cols());
    std::vector<boost::unordered_flat_map<id_type, int>> pos_maps(n_points);
    OMP_DIRECTIVE(parallel for)
    for (id_type sid = 0; sid < n_points; ++sid) {
      for (int i = 0; i < in_knng_ids.n_cols(); ++i) {
        detour_counts_table(sid, i)        = 0;
        pos_maps[sid][in_knng_ids(sid, i)] = i;
      }
    }

    // Count detours
    OMP_DIRECTIVE(parallel for)
    for (id_type sid = 0; sid < n_points; ++sid) {
      for (int i = 0; i < in_knng_ids.n_cols(); ++i) {
        // Transfer point
        const auto tid     = in_knng_ids(sid, i);
        const auto dist_st = in_knng_dists(sid, i);
        for (int j = 0; j < in_knng_ids.n_cols(); ++j) {
          const auto zid     = in_knng_ids(tid, j);
          const auto dist_tz = in_knng_dists(tid, j);
          // Check if there is a detour from sid to zid via tid,
          // i.e., dist(sid, zid) > dist(sid, tid) &&
          // dist(sid, zid) > dist(tid,zid)
          if (pos_maps[sid].count(zid) == 0) {
            continue;
          }
          const auto pos_zid = pos_maps[sid].at(zid);
          const auto dist_sz = in_knng_dists(sid, pos_zid);
          if (dist_sz > dist_st && dist_sz > dist_tz) {
            // Detour found
            ++detour_counts_table(sid, pos_zid);
          }
        }
      }
    }
    rec_time().stop();  // Count-detours

    spdlog::trace("Make ranked knng");
    rec_time().start("Make-ranked-knng");
    matrix<id_type> ranked_knng(n_points, in_knng_ids.n_cols());
    std::memcpy(ranked_knng.data(), in_knng_ids.data(), in_knng_ids.mem_size());
    // Sort neighbors by detour counts (ascending)
    // Less detourable neighbors come first
    OMP_DIRECTIVE(parallel for)
    for (id_type sid = 0; sid < n_points; ++sid) {
      std::sort(ranked_knng(sid), ranked_knng(sid) + ranked_knng.n_cols(),
                [&](const id_type a, const id_type b) {
                  const size_t count_a =
                      detour_counts_table(sid, pos_maps[sid].at(a));
                  const size_t count_b =
                      detour_counts_table(sid, pos_maps[sid].at(b));
                  return count_a < count_b;
                });
    }
    rec_time().stop();  // Make-ranked-knng
                        // Builds a query graph by ranking neighbors via detour
                        // counts and reverse links.
    pos_maps.clear();   // No longer needed
    detour_counts_table.reset();  // No longer needed

    spdlog::trace("Make reverse ranked knng");
    rec_time().start("Make-reverse-ranked-knng");
    // Make reverse ranked knng
    knn_heap_adj_list_t     r_ranked_knng_heap(n_points,
                                               knn_heap_t(out_query_nids.n_cols()));
    std::vector<std::mutex> mutexes(2048);
    OMP_DIRECTIVE(parallel for)
    for (id_type sid = 0; sid < ranked_knng.n_rows(); ++sid) {
      for (int i = 0; i < ranked_knng.n_cols(); ++i) {
        const auto tid = ranked_knng(sid, i);
        {
          std::lock_guard<std::mutex> lock(mutexes[tid % mutexes.size()]);
          r_ranked_knng_heap[tid].try_add(sid, i);
        }
      }
    }
    rec_time().stop();  // Make-reverse-ranked-knng

    spdlog::trace("Merge ranked knng and reverse ranked knng");
    rec_time().start("Merge-ranked-knngs");
    // Merge the half of the neighbors with the lowest detour counts in
    // ranked_knng and r_ranked_knng_heap
    OMP_DIRECTIVE(parallel for)
    for (id_type sid = 0; sid < ranked_knng.n_rows(); ++sid) {
      knn_heap_t knn_heap(out_query_nids.n_cols());
      assert(ranked_knng.n_cols() >= knn_heap.k() / 2);
      for (int i = 0; i < knn_heap.k() / 2; ++i) {
        knn_heap.try_add(ranked_knng(sid, i), i);
      }

      // Merge the reversed knng.
      // Assumes r_ranked_nbs is sorted by rank (ascending)
      const auto r_ranked_nbs = r_ranked_knng_heap[sid].extract_neighbors();
      for (int i = 0; i < std::min<int>(r_ranked_nbs.size(), knn_heap.k() / 2);
           ++i) {
        knn_heap.try_add(r_ranked_nbs[i].id, r_ranked_nbs[i].distance);
      }

      // Fill the rest of space with the neighbors in ranked_knng
      for (int i = knn_heap.k() / 2;
           i < ranked_knng.n_cols() &&
           knn_heap.size() <= static_cast<size_t>(knn_heap.k());
           ++i) {
        knn_heap.try_add(ranked_knng(sid, i), i);
      }
      assert(knn_heap.k() == static_cast<size_t>(out_query_nids.n_cols()));
      assert(knn_heap.size() == static_cast<size_t>(out_query_nids.n_cols()));

      // Write back to query graph
      const auto new_neighbors = knn_heap.extract_neighbors();
      if (new_neighbors.size() != out_query_nids.n_cols()) {
        std::cerr << "Size mismatch: new_neighbors.size() = "
                  << new_neighbors.size()
                  << ", expected = " << out_query_nids.n_cols() << std::endl;
      }
      assert(new_neighbors.size() ==
             static_cast<size_t>(out_query_nids.n_cols()));
      for (int i = 0; i < new_neighbors.size(); ++i) {
        out_query_nids(sid, i) = new_neighbors[i].id;
      }
    }
    rec_time().stop();  // Merge-ranked-knngs
}

template <typename id_type, typename dist_type>
[[deprecated("Use make_optimized_query_graph_apu.")]] inline void
make_opmized_query_graph_apu(const matrix_view<id_type>&     in_knng_ids,
                             const matrix_view<dist_type>&   in_knng_dists,
                             matrix_view<id_type>            out_query_nids,
                             const query_graph_distance_mode distance_mode =
                                 query_graph_distance_mode::actual_distance) {
  make_optimized_query_graph_apu<id_type, dist_type>(
      in_knng_ids, in_knng_dists, out_query_nids, distance_mode);
}

template <typename id_type, typename dist_type>
[[deprecated("Use make_optimized_query_graph.")]] inline void
make_opmized_query_graph(const matrix_view<id_type>&   in_knng_ids,
                         const matrix_view<dist_type>& in_knng_dists,
                         matrix_view<id_type>          out_query_nids) {
  make_optimized_query_graph<id_type, dist_type>(in_knng_ids, in_knng_dists,
                                                 out_query_nids);
}
}  // namespace saltatlas::solanet::apu_nn
