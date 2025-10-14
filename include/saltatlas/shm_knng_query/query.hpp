// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <boost/unordered/unordered_flat_set.hpp>
#include <metall/utility/random.hpp>

#include <saltatlas/common/detail/neighbor.hpp>
#include <saltatlas/common/detail/utilities/float.hpp>
#include <saltatlas/common/detail/utilities/general.hpp>
#include <saltatlas/common/point_store.hpp>
#include <saltatlas/dnnd/detail/nn_index.hpp>
#include <saltatlas/dnnd/detail/utilities/bitset.hpp>
#include <saltatlas/dnnd/detail/utilities/time.hpp>
#include "saltatlas/dnnd/detail/utilities/omp.hpp"
#include "saltatlas/dnnd/distance.hpp"

namespace saltatlas {

namespace {
namespace omp = saltatlas::utility::omp;
}

template <typename PointStore, typename NNIndex, typename IdType,
          typename DistanceType, typename FeatureElementType>
class knn_parallel_query_kernel {
 public:
  using id_type          = IdType;
  using distance_type    = DistanceType;
  using fe_type          = FeatureElementType;
  using point_store_type = PointStore;
  using nn_index_type    = NNIndex;

  using point_wrapper_type = std::span<fe_type>;
  using distance_function_type =
      distance::distance_function_type<point_wrapper_type, distance_type>;
  using neighbor_type = detail::neighbor<id_type, distance_type>;

  struct option {
    std::size_t k{0};
    double      epsilon{0.0};
    uint64_t    rnd_seed{128};
    bool        verbose{false};
  };

 private:
  // Descent ordered heap.
  // Longest element is at the top
  struct neighbor_less {
    bool operator()(const neighbor_type& lhd, const neighbor_type& rhd) const {
      return lhd.distance < rhd.distance;
    }
  };
  using dsc_heap_type =
      std::priority_queue<neighbor_type, std::vector<neighbor_type>,
                          neighbor_less>;

  // Ascent ordered heap.
  // Smallest element is at the top
  struct neighbor_greater {
    bool operator()(const neighbor_type& lhd, const neighbor_type& rhd) const {
      return lhd.distance > rhd.distance;
    }
  };
  using asc_heap_type =
      std::priority_queue<neighbor_type, std::vector<neighbor_type>,
                          neighbor_greater>;

 public:
  knn_parallel_query_kernel(const option&           opt,
                            const point_store_type& point_store,
                            const std::string_view& distance_function_name,
                            const nn_index_type&    nn_index,
                            const int               num_threads = -1)
      : m_option(opt),
        m_point_store(point_store),
        m_distance_function(
            distance::distance_function<point_wrapper_type, distance_type>(
                distance_function_name)),
        m_nn_index(nn_index) {
    if (num_threads > 0) {
      omp::set_num_threads(num_threads);
    }
    // OMP_DIRECTIVE(parallel) {
    //   OMP_DIRECTIVE(single)
    //   std::cout << "Will use " << omp::get_num_threads() << " threads"
    //             << std::endl;
    // }
  }

  std::vector<std::vector<neighbor_type>> query(
      const std::vector<std::vector<fe_type>>& queries) {
    std::vector<std::vector<neighbor_type>> results;

    OMP_DIRECTIVE(parallel) {
      const auto tid      = omp::get_thread_num();
      const auto nthreads = omp::get_num_threads();
      const auto range =
          saltatlas::detail::partial_range(queries.size(), tid, nthreads);
      metall::utility::rand_512 rnd_generator(m_option.rnd_seed);

      OMP_DIRECTIVE(single)
      results.resize(queries.size());

      OMP_DIRECTIVE(barrier)

      for (std::size_t i = range.first; i < range.second; ++i) {
        results[i] = query_single(queries[i], rnd_generator);
      }
    }
    return results;
  }

  template <typename RndGenerator>
  std::vector<neighbor_type> query_single(const std::vector<fe_type>& query,
                                          RndGenerator& rnd_generator) {
    asc_heap_type frontier;
    dsc_heap_type knn_heap;

    boost::unordered_flat_set<id_type> visited;

    // Initialize the frontier
    {
      const auto n = m_option.k;
      // std::max(std::size_t(1), std::size_t(std::sqrt(m_option.k)));
      for (std::size_t i = 0; i < n; ++i) {
        while (true) {
          std::uniform_int_distribution<> dis(0,
                                              m_point_store.num_points() - 1);
          const id_type                   id = dis(rnd_generator);
          if (visited.count(id)) continue;
          visited.insert(id);

          const auto d = m_distance_function(
              std::span(const_cast<fe_type*>(query.data()), query.size()),
              std::span(const_cast<fe_type*>(m_point_store.at(id)),
                        m_point_store.num_dimensions()));
          const auto candidate = neighbor_type(id, d);
          frontier.push(candidate);
          knn_heap.push(candidate);
          break;
        }
      }
    }

    const double distance_scale = 1.0 + m_option.epsilon;
    double       distance_bound = (knn_heap.size() >= m_option.k)
                                      ? knn_heap.top().distance * distance_scale
                                      : std::numeric_limits<double>::max();

    // Main search loop
    {
      while (!frontier.empty()) {
        const auto source = frontier.top();  // Search from the closest point
        frontier.pop();

        // As source is the closest point in the frontier,
        // all points in the frontier is farther than distance_bound if the
        // following condition is true.
        if (source.distance > distance_bound) break;

        for (auto [nitr, nend] = m_nn_index.neighbors(source.id); nitr != nend;
             ++nitr) {
          const auto nid = *nitr;
          if (visited.count(nid)) continue;
          visited.insert(nid);

          const auto d = m_distance_function(
              std::span(const_cast<fe_type*>(query.data()), query.size()),
              std::span(const_cast<fe_type*>(m_point_store.at(nid)),
                        m_point_store.num_dimensions()));
          if (d >= distance_bound) continue;

          const auto candidate = neighbor_type(nid, d);
          frontier.push(candidate);

          // Update knn_heap
          if (knn_heap.size() >= m_option.k &&
              knn_heap.top().distance > candidate.distance) {
            knn_heap.pop();
          }
          if (knn_heap.size() < m_option.k) {
            knn_heap.push(candidate);
            distance_bound = knn_heap.top().distance * distance_scale;
          }
        }
      }
    }

    std::vector<neighbor_type> result;
    {
      result.reserve(m_option.k);
      while (!knn_heap.empty()) {
        result.push_back(knn_heap.top());
        knn_heap.pop();
      }
      std::reverse(result.begin(), result.end());
    }

    return result;
  }

 private:
  const option            m_option;
  const point_store_type& m_point_store;
  distance_function_type  m_distance_function;
  const nn_index_type&    m_nn_index;
};

}  // namespace saltatlas