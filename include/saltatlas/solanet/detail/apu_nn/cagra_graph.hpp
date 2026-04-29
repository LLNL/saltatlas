// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <cassert>
#include <cstddef>
#include <mutex>
#include <vector>

#include <spdlog/spdlog.h>

#include <boost/unordered/unordered_flat_map.hpp>

#include "saltatlas/common/detail/neighbor.hpp"
#include "saltatlas/common/detail/utilities/general.hpp"
#include "saltatlas/dnnd/detail/knn_heap.hpp"
#include "saltatlas/dnnd/detail/utilities/omp.hpp"
#include "saltatlas/solanet/detail/apu_nn/matrix.hpp"
#include "saltatlas/solanet/detail/nn_index_view.hpp"
#include "saltatlas/solanet/detail/utilities/mutex.hpp"
#include "saltatlas/solanet/singleton_time_recorder.hpp"
#include "saltatlas/neo_dnnd/mpi.hpp"

namespace saltatlas::solanet::apu_nn {
namespace {
namespace omp   = saltatlas::utility::omp;
namespace mutex = saltatlas::dndetail::mutex;
namespace bst   = boost;
namespace bstuo = boost::unordered;
}  // namespace

// CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor
// Search for GPUs
template <typename id_type>
inline void build_cagra_graph(matrix_view<id_type> knng,
                              matrix_view<id_type> cagra_knng) {
  const size_t n_points     = knng.n_rows();
  const size_t input_degree = knng.n_cols();
  assert(input_degree >= cagra_knng.n_cols());

  rec_time().start("Count-detours");
  // Count detours from s to t via u:
  // There is a detour from s to t via u if
  // Rank(s->t) > Rank(s->u) && Rank(s->t) > Rank(u->t),
  // where Rank(x->y) is the rank of y in x's neighbor list.
  matrix<size_t> detour_counts_table(n_points, input_degree);
  OMP_DIRECTIVE(parallel for)
  for (size_t sid = 0; sid < n_points; ++sid) {
    bstuo::unordered_flat_map<id_type, size_t> t_pos_map;
    for (size_t pos = 0; pos < input_degree; ++pos) {
      const auto t                  = knng(sid, pos);
      t_pos_map[t]                  = pos;
      detour_counts_table(sid, pos) = 0;
    }

    for (size_t i = 0; i < input_degree; ++i) {
      const auto u = knng(sid, i);
      for (size_t j = 0; j < input_degree; ++j) {
        const auto t = knng(u, j);
        if (t_pos_map.count(t) > 0) {
          const int rank_su = i;
          const int rank_ut = j;
          const int pos_t   = t_pos_map.at(t);
          const int rank_st = pos_t;
          if (rank_st > rank_su && rank_st > rank_ut) {
            // Detour found
            ++detour_counts_table(sid, pos_t);
          }
        }
      }
    }

    // Sort neighbors by detour counts (ascending)
    // Less detourable neighbors come first
    std::sort(
        knng(sid), knng(sid) + input_degree,
        [&](const id_type a, const id_type b) {
          const size_t count_a = detour_counts_table(sid, t_pos_map.at(a));
          const size_t count_b = detour_counts_table(sid, t_pos_map.at(b));
          return count_a < count_b;
        });
  }
  rec_time().stop();

  // Build reverse kNNG
  // Each point holds up to 'cagra_knng.n_cols() / 2' neighbors with the
  // lowest detour counts
  using knn_heap_t          = dndetail::unique_knn_heap<id_type, int>;
  using knn_heap_adj_list_t = std::vector<knn_heap_t>;

  rec_time().start("Build-reverse-knng");
  // TODO: Optimize initialization of r_knng
  knn_heap_adj_list_t     r_knng(n_points, knn_heap_t(cagra_knng.n_cols() / 2));
  std::vector<std::mutex> mutexes(1024);

  OMP_DIRECTIVE(parallel for)
  for (size_t sid = 0; sid < n_points; ++sid) {
    for (size_t pos = 0; pos < input_degree; ++pos) {
      const auto t = knng(sid, pos);
      // Insert (s, pos) into r_knng[t]
      std::lock_guard<std::mutex> lock(mutexes[t % mutexes.size()]);
      r_knng[t].try_add(sid, static_cast<int>(pos));
    }
  }
  rec_time().stop();

  rec_time().start("Merge-knngs-for-cagra");
  OMP_DIRECTIVE(parallel for)
  for (size_t sid = 0; sid < n_points; ++sid) {
    auto& knn_heap = r_knng[sid];
    while (knn_heap.size() > cagra_knng.n_cols() / 2) {
      knn_heap.pop();
    }
    // Insert remaining neighbors into knng
    int pos = cagra_knng.n_cols() - 1;
    while (!knn_heap.empty()) {
      const auto neighbor  = knn_heap.top();
      cagra_knng(sid, pos) = neighbor.id;
      knn_heap.pop();
      --pos;
    }
    // Fill the rest of knng with original neighbors
    for (; pos >= 0; --pos) {
      cagra_knng(sid, pos) = knng(sid, pos);
    }
  }
  rec_time().stop();
}
}  // namespace saltatlas::solanet::apu_nn