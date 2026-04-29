// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace saltatlas::solanet::d3dtl {

template <typename IDType, typename FeatureElemType>
class point_store_view {
 public:
  using id_type          = IDType;
  using fe_type          = FeatureElemType;
  using point_type       = std::span<fe_type>;
  using const_point_type = std::span<const fe_type>;

  point_store_view() = default;

  // Construct from serialized buffer
  point_store_view(const size_t num_points, const size_t num_dims,
                   void *const buf) {
    priv_init_with_not_owned_data(num_points, num_dims,
                                  static_cast<fe_type *>(buf));
  }

  ~point_store_view() noexcept { reset(); }

  // Copy constructor
  point_store_view(const point_store_view &) = default;

  // Copy assignment
  point_store_view &operator=(const point_store_view &) = default;

  // Move constructor
  point_store_view(point_store_view &&other) noexcept
      : m_num_points(other.m_num_points),
        m_num_dims(other.m_num_dims),
        m_data(other.m_data) {
    other.m_num_points = 0;
    other.m_num_dims   = 0;
    other.m_data       = nullptr;
  }

  // Move assignment
  point_store_view &operator=(point_store_view &&other) noexcept {
    if (this != &other) {
      m_num_points       = other.m_num_points;
      m_num_dims         = other.m_num_dims;
      m_data             = other.m_data;
      other.m_num_points = 0;
      other.m_num_dims   = 0;
      other.m_data       = nullptr;
    }
    return *this;
  }

  const_point_type operator[](id_type id) const {
#ifndef NDEBUG
    if (id >= m_num_points) {
      std::cerr << __FILE__ << " " << __LINE__ << ": ID is out of range: " << id
                << " vs " << m_num_points << std::endl;
      std::abort();
    }
#endif
    if (!m_data) {
      std::cerr << __FILE__ << " " << __LINE__ << ": Data pointer is null."
                << std::endl;
      std::abort();
    }
    return const_point_type(m_data + id * m_num_dims, m_num_dims);
  }

  point_type operator[](id_type id) {
#ifndef NDEBUG
    if (id >= m_num_points) {
      std::cerr << __FILE__ << " " << __LINE__ << ": ID is out of range: " << id
                << " vs " << m_num_points << std::endl;
      std::abort();
    }
#endif
    if (!m_data) {
      std::cerr << __FILE__ << " " << __LINE__ << ": Data pointer is null."
                << std::endl;
      std::abort();
    }
    return point_type(m_data + id * m_num_dims, m_num_dims);
  }

  std::size_t size() const { return m_num_points; }

  std::size_t dims() const { return m_num_dims; }

  // Serialize the data
  fe_type *data() { return m_data; }

  const fe_type *data() const { return m_data; }

  void reset(const size_t num_points = 0, const size_t num_dims = 0,
             void *const buf = nullptr) {
    m_num_points = num_points;
    m_num_dims   = num_dims;
    m_data       = static_cast<fe_type *>(buf);
  }

 private:
  void priv_init_with_not_owned_data(std::size_t    num_points,
                                     std::size_t    num_dims,
                                     fe_type *const data) {
    m_num_points = num_points;
    m_num_dims   = num_dims;
    m_data       = data;
  }

  std::size_t m_num_points{0};
  std::size_t m_num_dims{0};
  fe_type    *m_data{nullptr};
};
}  // namespace saltatlas::solanet::d3dtl