// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <random>
#include <string_view>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/detail/base_dnnd.hpp>

namespace saltatlas {

/// \brief Distributed NNDescent.
/// \tparam Id Point ID type.
/// \tparam FeatureElement Feature vector element type.
/// \tparam Distance Distance type.
template <typename Id = uint64_t, typename FeatureElement = double,
          typename Distance = double>
class dnnd : public dndetail::base_dnnd<Id, FeatureElement, Distance> {
 private:
  using base_type      = dndetail::base_dnnd<Id, FeatureElement, Distance>;
  using data_core_type = typename base_type::data_core_type;

 public:
  using id_type              = typename base_type::id_type;
  using feature_element_type = typename base_type::feature_element_type;
  using distance_type        = typename base_type::distance_type;
  using point_store_type     = typename base_type::point_store_type;
  using knn_index_type       = typename base_type::knn_index_type;
  using feature_vector_type  = typename base_type::feature_vector_type;
  using neighbor_type        = typename base_type::neighbor_type;
  using query_store_type     = typename base_type::query_store_type;
  using point_partitioner    = typename base_type::point_partitioner;
  using neighbor_store_type  = typename base_type::neighbor_store_type;

  /// \brief Constructor.
  /// \param distance_metric_name Distance metric name.
  /// \param comm YGM comm instance.
  /// \param rnd_seed Seed for random generators.
  /// \param verbose If true, enable the verbose mode.
  dnnd(const std::string_view distance_metric_name, ygm::comm& comm,
       const uint64_t rnd_seed = std::random_device{}(),
       const bool     verbose  = false)
      : base_type(verbose, comm),
        m_data_core(
            dndetail::distance::convert_to_metric_id(distance_metric_name),
            rnd_seed),
        m_comm(comm) {
    base_type::init_data_core(m_data_core);
  }

 private:
  data_core_type m_data_core;
  ygm::comm&     m_comm;
};

}  // namespace saltatlas
