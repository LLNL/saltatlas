// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <boost/container/vector.hpp>

#include <saltatlas/common/detail/utilities/string_cast.hpp>
#include <saltatlas/dnnd/detail/utilities/omp.hpp>

namespace saltatlas {

namespace {
namespace omp = saltatlas::utility::omp;
}

template <typename Id, typename T, typename Alloc = std::allocator<T>>
class dense_point_store {
  static_assert(
      std::is_same_v<T, typename std::allocator_traits<Alloc>::value_type>,
      "Alloc::value_type must be the same as T");

 private:
  using self_type    = dense_point_store<Id, T, Alloc>;
  using pointer_type = typename std::allocator_traits<Alloc>::pointer;

 public:
  using id_type        = Id;
  using value_type     = T;
  using allocator_type = Alloc;

  dense_point_store(const std::size_t num_points, const std::size_t dims,
                    const allocator_type& alloc = allocator_type{})
      : m_num_points(num_points), m_dims(dims), m_allocator(alloc) {
    priv_alloc();
  }

  ~dense_point_store() { priv_dealloc(); }

  dense_point_store(const dense_point_store& other)      = delete;
  dense_point_store& operator=(const dense_point_store&) = delete;

  dense_point_store(dense_point_store&& other) noexcept { swap(other); }

  dense_point_store& operator=(dense_point_store&& other) noexcept {
    swap(other);
    return *this;
  }

  void swap(self_type& other) noexcept {
    std::swap(m_allocator, other.m_allocator);
    std::swap(m_dims, other.m_dims);
    std::swap(m_num_points, other.m_num_points);
    std::swap(m_data, other.m_data);
  }

  T* data() { return m_data; }

  const T* data() const { return m_data; }

  T* at(const std::size_t pid) {
    return const_cast<T*>(const_cast<const dense_point_store*>(this)->at(pid));
  }

  const T* at(const std::size_t pid) const {
    assert(pid < m_num_points);
    return std::to_address(m_data) + pid * m_dims;
  }

  void assign(const std::size_t pid, const std::vector<T>& point) {
    if (pid >= m_num_points) {
      std::cerr << "pid is out of range [0, " << m_num_points << ")"
                << std::endl;
      std::abort();
    }
    if (point.size() != m_dims) {
      std::cerr << "Unexpected #of dimensions." << std::endl;
      std::cerr << "Expected: " << m_dims << ", read: " << point.size()
                << std::endl;
      std::abort();
    }
    std::copy(point.begin(), point.end(),
              std::to_address(m_data) + pid * m_dims);
  }

  std::size_t num_dimensions() const { return m_dims; }

  std::size_t num_points() const { return m_num_points; }

 private:
  void priv_alloc() {
    assert(!m_data);
    assert(m_num_points >= 0);
    assert(m_dims > 0);
    const std::size_t total_elements = m_dims * m_num_points;
    m_data = std::allocator_traits<allocator_type>::allocate(m_allocator,
                                                             total_elements);
    assert(m_data);
  }

  void priv_dealloc() {
    if (m_data) {
      std::allocator_traits<allocator_type>::deallocate(m_allocator, m_data,
                                                        m_dims * m_num_points);
      m_data       = nullptr;
      m_num_points = 0;
      m_dims       = 0;
    }
  }

  allocator_type m_allocator;
  std::size_t    m_dims{0};
  std::size_t    m_num_points{0};
  pointer_type   m_data{nullptr};
};

}  // namespace saltatlas