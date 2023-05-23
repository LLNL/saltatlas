// Copyright 2023 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <random>

#include <saltatlas/dnnd/detail/distance.hpp>
#include <saltatlas/dnnd/detail/feature_vector.hpp>
#include <saltatlas/dnnd/detail/neighbor.hpp>
#include <saltatlas/dnnd/detail/nn_index.hpp>
#include <saltatlas/dnnd/detail/point_store.hpp>
#include <saltatlas/dnnd/detail/utilities/bitset.hpp>
#include <saltatlas/dnnd/detail/utilities/float.hpp>
#include <saltatlas/dnnd/detail/utilities/time.hpp>

#ifndef SALTATLAS_NND_QUERY_STAT_DETAIL
#define SALTATLAS_NND_QUERY_STAT_DETAIL 0
#endif

namespace saltatlas {

template <typename PointStore, typename NNIndex, typename IdType,
          typename DistanceType, typename FeatureElementType>
class knn_query_kernel {
 private:
  struct stat {
    std::size_t num_visited{0};
    std::size_t num_queries{0};
    double      total_init_time{0.0};
    double      total_query_time{0.0};
    double      total_convert_time{0.0};

    void reset() {
      num_visited        = 0;
      num_queries        = 0;
      total_init_time    = 0.0;
      total_query_time   = 0.0;
      total_convert_time = 0.0;
    }
  };

 public:
  using id_type              = IdType;
  using distance_type        = DistanceType;
  using feature_element_type = FeatureElementType;
  using point_store_type     = PointStore;
  using nn_index_type        = NNIndex;

  using distance_metric =
      dndetail::distance::metric_type<feature_element_type, distance_type>;
  using neighbor_type = dndetail::neighbor<id_type, distance_type>;

  struct option {
    int      k{0};
    double   epsilon{0.0};
    uint64_t rnd_seed{128};
    bool     verbose{false};
  };

  // Descent ordered heap.
  // Longest element is at the top
  using dsc_heap_type =
      std::priority_queue<neighbor_type, std::vector<neighbor_type>>;

  // Ascent ordered heap.
  // Smallest element is at the top
  struct neighbor_greater {
    bool operator()(const neighbor_type& lhd, const neighbor_type& rhd) const {
      if (lhd.distance == rhd.distance) return false;
      return !(lhd.distance < rhd.distance);
    }
  };
  using asc_heap_type =
      std::priority_queue<neighbor_type, std::vector<neighbor_type>,
                          neighbor_greater>;

  knn_query_kernel(const option& opt, const point_store_type& point_store,
                   const std::string_view& distance_metric_name,
                   const nn_index_type&    nn_index)
      : m_option(opt),
        m_point_store(point_store),
        m_distance_metric(
            dndetail::distance::metric<feature_element_type, distance_type>(
                distance_metric_name)),
        m_nn_index(nn_index),
        m_rnd_generator(m_option.rnd_seed),
        m_visited(m_point_store.size()) {
    reset_stat();
  }

  void reset_stat() { m_stat.reset(); }

  void show_stats() const {
    std::cout << "Mean #of visited points: "
              << static_cast<double>(m_stat.num_visited) /
                     static_cast<double>(m_stat.num_queries)
              << std::endl;
#if SALTATLAS_NND_QUERY_STAT_DETAIL
    std::cout << "Mean init time (s): "
              << m_stat.total_init_time / m_stat.num_queries << std::endl;
    std::cout << "Mean query core time (s): "
              << m_stat.total_query_time / m_stat.num_queries << std::endl;
    std::cout << "Mean convert time (s): "
              << m_stat.total_convert_time / m_stat.num_queries << std::endl;
#endif
  }

  std::vector<neighbor_type> query(
      const std::vector<feature_element_type>& query) {
    ++m_stat.num_queries;

    asc_heap_type frontier;
    dsc_heap_type knn_heap;

    {
#if SALTATLAS_NND_QUERY_STAT_DETAIL
      const auto start = dndetail::get_time();
#endif

      m_visited.reset_all();

      const auto n =
          std::max(std::size_t(1), std::size_t(std::sqrt(m_option.k)));
      for (std::size_t i = 0; i < n; ++i) {
        while (true) {
          std::uniform_int_distribution<> dis(0, m_point_store.size() - 1);
          const id_type                   id = dis(m_rnd_generator);
          if (m_visited.get(id)) continue;
          m_visited.set(id);
          const auto& feature   = m_point_store.at(id);
          const auto  d         = m_distance_metric(query.data(), query.size(),
                                                    feature.data(), feature.size());
          const auto  candidate = neighbor_type(id, d);
          frontier.push(candidate);
          knn_heap.push(candidate);
          break;
        }
        ++m_stat.num_visited;
      }
#if SALTATLAS_NND_QUERY_STAT_DETAIL
      const auto t = dndetail::elapsed_time_sec(start);
      m_stat.total_init_time += t;
#endif
    }

    const double distance_scale = 1.0 + m_option.epsilon;
    double       distance_bound = (knn_heap.size() >= m_option.k)
                                      ? knn_heap.top().distance * distance_scale
                                      : std::numeric_limits<double>::max();

    {
#if SALTATLAS_NND_QUERY_STAT_DETAIL
      const auto start = dndetail::get_time();
#endif
      // Main search loop
      while (!frontier.empty()) {
        const auto source = frontier.top();  // Search from the closest point
        frontier.pop();

        ++m_stat.num_visited;

        // As source is the closest point in the frontier,
        // all points in the frontier is farther than distance_bound if the
        // following condition is true.
        if (source.distance > distance_bound) break;

        for (const auto& nid : m_nn_index.at(source.id)) {
          if (m_visited.get(nid)) continue;
          m_visited.set(nid);

          const auto& feature = m_point_store.at(nid);
          const auto  d       = m_distance_metric(query.data(), query.size(),
                                                  feature.data(), feature.size());
          if (d >= distance_bound) continue;

          const auto candidate = neighbor_type(nid, d);
          frontier.push(candidate);
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
#if SALTATLAS_NND_QUERY_STAT_DETAIL
      const auto t = dndetail::elapsed_time_sec(start);
      m_stat.total_query_time += t;
#endif
    }

    std::vector<neighbor_type> result;
    {
#if SALTATLAS_NND_QUERY_STAT_DETAIL
      const auto start = dndetail::get_time();
#endif
      result.reserve(m_option.k);
      while (!knn_heap.empty()) {
        result.push_back(knn_heap.top());
        knn_heap.pop();
      }
      std::reverse(result.begin(), result.end());
#if SALTATLAS_NND_QUERY_STAT_DETAIL
      const auto t = dndetail::elapsed_time_sec(start);
      m_stat.total_convert_time += t;
#endif
    }

    return result;
  }

 private:
  option                              m_option;
  const point_store_type&             m_point_store;
  const distance_metric&              m_distance_metric;
  const nn_index_type&                m_nn_index;
  std::mt19937                        m_rnd_generator;
  saltatlas::dndetail::dynamic_bitset m_visited;
  stat                                m_stat;
};

}  // namespace saltatlas