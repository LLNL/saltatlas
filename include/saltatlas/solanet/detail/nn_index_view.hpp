// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

namespace saltatlas::solanet::d3dtl {

template <typename IDType, typename DistanceType>
class nn_index_view {
 public:
  using id_type   = IDType;
  using dist_type = DistanceType;

  nn_index_view() noexcept = default;

  nn_index_view(const size_t num_points, const size_t num_neighbors,
                void* const ids, void* const dists)
      : m_num_points(num_points),
        m_num_neighbors(num_neighbors),
        m_id_data(static_cast<id_type*>(ids)),
        m_dist_data(static_cast<dist_type*>(dists)) {}

  // Destructor - free owned data
  ~nn_index_view() noexcept { reset(); }

  nn_index_view(const nn_index_view&)            = default;
  nn_index_view& operator=(const nn_index_view&) = default;

  // Move constructor
  nn_index_view(nn_index_view&& other) noexcept
      : m_num_points(std::move(other.m_num_points)),
        m_num_neighbors(std::move(other.m_num_neighbors)),
        m_id_data(std::move(other.m_id_data)),
        m_dist_data(std::move(other.m_dist_data)) {
    other.m_num_points    = 0;
    other.m_num_neighbors = 0;
    other.m_id_data       = nullptr;
    other.m_dist_data     = nullptr;
  }

  // Move assignment
  nn_index_view& operator=(nn_index_view&& other) noexcept {
    if (this != &other) {
      m_num_points    = std::move(other.m_num_points);
      m_num_neighbors = std::move(other.m_num_neighbors);
      m_id_data       = std::move(other.m_id_data);
      m_dist_data     = std::move(other.m_dist_data);

      other.m_num_points    = 0;
      other.m_num_neighbors = 0;
      other.m_id_data       = nullptr;
      other.m_dist_data     = nullptr;
    }
    return *this;
  }

  std::span<id_type> neighbor_ids(const id_type id) {
    priv_check_id_range(id);
    return std::span<id_type>(
        m_id_data + static_cast<std::size_t>(id) * m_num_neighbors,
        m_num_neighbors);
  }

  std::span<const id_type> neighbor_ids(const id_type id) const {
    priv_check_id_range(id);
    return std::span<const id_type>(
        m_id_data + static_cast<std::size_t>(id) * m_num_neighbors,
        m_num_neighbors);
  }

  std::span<dist_type> neighbor_dists(const id_type id) {
    priv_check_id_range(id);
    return std::span<dist_type>(
        m_dist_data + static_cast<std::size_t>(id) * m_num_neighbors,
        m_num_neighbors);
  }

  std::span<const dist_type> neighbor_dists(const id_type id) const {
    priv_check_id_range(id);
    return std::span<const dist_type>(
        m_dist_data + static_cast<std::size_t>(id) * m_num_neighbors,
        m_num_neighbors);
  }

  inline std::size_t size() const { return m_num_points; }

  inline std::size_t num_neighbors() const { return m_num_neighbors; }

  // Returns reference to the internal id buffer.
  // Also returns size in bytes.
  id_type* neighbor_ids_data() {
    if (!m_id_data) {
      return nullptr;
    }
    return m_id_data;
  }

  const id_type* neighbor_ids_data() const {
    if (!m_id_data) {
      return nullptr;
    }
    return m_id_data;
  }

  // Returns reference to the internal distance buffer.
  // Also returns size in bytes.
  dist_type* dists_data() {
    if (!m_dist_data) {
      return nullptr;
    }
    return m_dist_data;
  }

  const dist_type* dists_data() const {
    if (!m_dist_data) {
      return nullptr;
    }
    return m_dist_data;
  }

  void reset(const size_t num_points = 0, const size_t num_neighbors = 0,
             void* const ids = nullptr, void* const dists = nullptr) {
    m_num_points    = num_points;
    m_num_neighbors = num_neighbors;
    m_id_data       = static_cast<id_type*>(ids);
    m_dist_data     = static_cast<dist_type*>(dists);
  }

  void* data() const { return static_cast<void*>(m_id_data); }

 private:
  inline void priv_check_id_range(const id_type id) const {
#ifndef NDEBUG
    if (static_cast<std::size_t>(id) >= m_num_points) {
      std::cerr << __FILE__ << " " << __LINE__ << ": ID is out of range: " << id
                << ". Expecting < " << m_num_points << std::endl;
      std::abort();
    }
#endif
  }

  inline void priv_check_number_of_neighbors(const std::size_t n) const {
#ifndef NDEBUG
    if (n >= m_num_neighbors) {
      std::cerr << __FILE__ << " " << __LINE__
                << ": neighbor index out of range: " << n << " vs "
                << m_num_neighbors << std::endl;
      std::abort();
    }
#endif
  }

  std::size_t m_num_points{0};
  std::size_t m_num_neighbors{0};
  id_type*    m_id_data{nullptr};
  dist_type*  m_dist_data{nullptr};
};
}  // namespace saltatlas::solanet::d3dtl