// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <random>
#include <string_view>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/detail/base_dnnd.hpp>
#include <saltatlas/dnnd/feature_vector.hpp>

namespace saltatlas {

/// \brief Distributed NNDescent.
/// \tparam Id Point ID type.
/// \tparam Point Point type.
/// \tparam Distance Distance type.
template <typename Id       = uint64_t,
          typename Point    = saltatlas::feature_vector<double>,
          typename Distance = double>
class dnnd : public dndetail::base_dnnd<Id, Point, Distance> {
 private:
  using base_type      = dndetail::base_dnnd<Id, Point, Distance>;
  using data_core_type = typename base_type::data_core_type;

 public:
  using id_type                = typename base_type::id_type;
  using distance_type          = typename base_type::distance_type;
  using point_store_type       = typename base_type::point_store_type;
  using knn_index_type         = typename base_type::knn_index_type;
  using point_type             = typename base_type::point_type;
  using neighbor_type          = typename base_type::neighbor_type;
  using query_store_type       = typename base_type::query_store_type;
  using point_partitioner      = typename base_type::point_partitioner;
  using neighbor_store_type    = typename base_type::neighbor_store_type;
  using distance_function_type = typename base_type::distance_function_type;

  /// \brief Constructor.
  /// \param distance_name Distance metric name.
  /// \param comm YGM comm instance.
  /// \param rnd_seed Seed for random generators.
  /// \param verbose If true, enable the verbose mode.
  dnnd(const std::string_view distance_name, ygm::comm& comm,
       const uint64_t rnd_seed = std::random_device{}(),
       const bool     verbose  = false)
      : base_type(verbose, comm),
        m_data_core(distance::convert_to_distance_id(distance_name), rnd_seed) {
    base_type::init_data_core(m_data_core);
  }

  /// \brief Constructor.
  /// \param distance_func Distance function.
  /// \param comm YGM comm instance.
  /// \param rnd_seed Seed for random generators.
  /// \param verbose If true, enable the verbose mode.
  dnnd(const distance_function_type& distance_func, ygm::comm& comm,
       const uint64_t rnd_seed = std::random_device{}(),
       const bool     verbose  = false)
      : base_type(verbose, comm), m_data_core(distance::id::custom, rnd_seed) {
    base_type::init_data_core(m_data_core, distance_func);
  }

 private:
  data_core_type m_data_core;
};

}  // namespace saltatlas
