// Copyright 2023 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <random>

#include <saltatlas/dnnd/detail/distance.hpp>
#include <saltatlas/dnnd/detail/feature_vector.hpp>
#include <saltatlas/dnnd/detail/neighbor.hpp>
#include <saltatlas/dnnd/detail/nn_index.hpp>
#include <saltatlas/dnnd/detail/point_store.hpp>
#include <saltatlas/dnnd/detail/utilities/float.hpp>

namespace saltatlas {

template <typename PointStore, typename NNIndex, typename IdType,
          typename DistanceType, typename FeatureElementType>
class knn_query_kernel {
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
        m_rnd_generator(m_option.rnd_seed) {}

  std::vector<neighbor_type> query(
      const std::vector<feature_element_type>& query) {
    asc_heap_type     frontier;
    dsc_heap_type     knn_heap;
    std::vector<bool> checked;
    checked.resize(m_point_store.size(), false);

    for (std::size_t i = 0; i < m_option.k; ++i) {
      while (true) {
        std::uniform_int_distribution<> dis(0, m_point_store.size() - 1);
        const id_type                   id = dis(m_rnd_generator);
        if (checked.at(id)) continue;
        checked.at(id)        = true;
        const auto& feature   = m_point_store.at(id);
        const auto  d         = m_distance_metric(query.data(), query.size(),
                                                  feature.data(), feature.size());
        const auto  candidate = neighbor_type(id, d);
        frontier.push(candidate);
        knn_heap.push(candidate);
        //++m_num_checked;
        break;
      }
    }

    const double distance_scale = 1.0 + m_option.epsilon;
    double       distance_bound = knn_heap.top().distance * distance_scale;

    // Main search loop
    while (!frontier.empty()) {
      const auto source = frontier.top();  // Search from the closest point
      frontier.pop();

      // As source is the closest point in the frontier,
      // all points in the frontier is farther than distance_bound if the
      // following condition is true.
      if (source.distance > distance_bound) break;

      for (const auto& nid : m_nn_index.at(source.id)) {
        if (checked.at(nid)) continue;
        checked.at(nid) = true;
        //++m_num_checked;

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
    //++m_num_processed_queries;

    std::vector<neighbor_type> result;
    result.reserve(m_option.k);
    while (!knn_heap.empty()) {
      result.push_back(knn_heap.top());
      knn_heap.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
  }

  option                  m_option;
  const point_store_type& m_point_store;
  const distance_metric&  m_distance_metric;
  const nn_index_type&    m_nn_index;
  std::mt19937            m_rnd_generator;
};

}  // namespace saltatlas