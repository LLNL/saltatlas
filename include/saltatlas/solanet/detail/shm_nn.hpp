// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <iostream>

namespace saltatlas::solanet {

template <typename IDType, typename FeatureElemType>
class dummpy_point_store {
 public:
  using id_type = IDType;
  using fe_type = FeatureElemType;

  dummpy_point_store() = default;
  dummpy_point_store(const std::byte* const);

  explicit dummpy_point_store(const dummpy_point_store&)     = default;
  explicit dummpy_point_store(dummpy_point_store&&) noexcept = default;

  // Construct from serialized buffer
  dummpy_point_store(std::byte* const buf);

  // Access a single point (pointer to first feature of point `idx`)
  fe_type*       operator[](std::size_t idx) noexcept { return nullptr; }
  const fe_type* operator[](std::size_t idx) const noexcept { return nullptr; }

  std::size_t size() const noexcept { return 0; }
  std::size_t dims() const noexcept { return 0; }

  fe_type*       data() noexcept { return nullptr; }
  const fe_type* data() const noexcept { return nullptr; }

  // Serialize the point store
  std::pair<const std::byte*, std::size_t> serialize() const { return {}; }

 private:
};

template <typename IDType, typename DistanceType>
class dummy_nn_index {
 public:
  using id_type   = IDType;
  using dist_type = DistanceType;

  dummy_nn_index() = default;
  dummy_nn_index(const std::byte* const);

  const id_type neigbor_id(const id_type id, const std::size_t n) const;
  id_type       neigbor_id(const id_type id, const std::size_t n);

  const dist_type distance(const id_type id, const std::size_t n) const;
  dist_type       distance(const id_type id, const std::size_t n);

  std::pair<id_type*, dist_type*>             data();
  std::pair<const id_type*, const dist_type*> data() const;

  std::pair<id_type*, dist_type*> to_host() const;
  std::pair<id_type*, dist_type*> to_dev() const;

  std::size_t                              size() const;
  std::pair<const std::byte*, std::size_t> serialize() const { return {}; }
};

template <typename IDType, typename DistanceType>
class dummy_query_result_type {
 public:
  using id_type   = IDType;
  using dist_type = DistanceType;

  const id_type  neigbor_id(const id_type id, const std::size_t n) const;
  id_type        neigbor_id(const id_type id, const std::size_t n);
  const id_type* neighbor_id_data(const id_type id) const;
  id_type*       neighbor_id_data(const id_type id);

  const dist_type  distance(const id_type id, const std::size_t n) const;
  dist_type        distance(const id_type id, const std::size_t n);
  const dist_type* distance_data(const id_type id) const;
  dist_type*       distance_data(const id_type id);

  void add_id_offset(const id_type offset);

  std::size_t size() const;
};

template <typename IDType, typename FeatureElemType, typename DistanceType>
class dummy_shm_nn_driver {
 public:
  using pstore_type = dummpy_point_store<IDType, FeatureElemType>;

  using index_type = dummy_nn_index<IDType, DistanceType>;

  using query_store_type  = pstore_type;
  using query_result_type = dummy_query_result_type<IDType, DistanceType>;

  using id_type   = IDType;
  using dist_type = DistanceType;
  using fe_type   = FeatureElemType;

  dummy_shm_nn_driver(const std::string_view dist_name);

  pstore_type allocate_pstore(const std::size_t n_points,
                              const std::size_t dims) {
    return {};
  }

  template <typename... Args>
  index_type build_index(const pstore_type& pstore, Args&&... /*args*/) {
    (void)sizeof...(Args);
    return {};
  }

  template <typename... Args>
  query_result_type search(const index_type&       index,
                           const query_store_type& query, Args&&... /*args*/) {
    (void)sizeof...(Args);
    return {};
  }

  std::size_t update_index(const query_result_type& candidates,
                           index_type&              index) {
    return 0;
  }
};

}  // namespace saltatlas::solanet
