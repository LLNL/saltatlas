// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstdint>
#include <memory>

#include <boost/version.hpp>
#include <metall/metall.hpp>
#if defined(BOOST_VERSION) && BOOST_VERSION >= 108700
#include <boost/unordered/unordered_flat_map.hpp>
#else
#error "Boost 1.87.00 or higher is required."
#endif

#include "saltatlas/dnnd/detail/utilities/allocator.hpp"

namespace saltatlas {

/// \brief Dataset store. There are N points in the dataset, each point is
/// represented by a feature vector. Every point has the same dimension.
template <typename _id_type, typename _value_type,
          typename _allocator_type = std::allocator<_value_type>>
class compact_point_store {
 public:
  using id_type        = _id_type;
  using value_type     = _value_type;
  using allocator_type = typename std::allocator_traits<
      _allocator_type>::template rebind_alloc<value_type>;
  using allocator_traits = std::allocator_traits<allocator_type>;

 private:
  using pointer = typename std::pointer_traits<
      typename allocator_traits::pointer>::template rebind<value_type>;

  /// Map from external ID to internal ID.
  using id_table_t = boost::unordered::unordered_flat_map<
      id_type, id_type, saltatlas::hash<13149>, std::equal_to<>,
      typename std::allocator_traits<allocator_type>::template rebind_alloc<
          std::pair<const id_type, id_type>>>;

 public:
  class const_id_iterator;
  class const_iterator;

  /// \brief Constructor.
  explicit compact_point_store(
      const allocator_type &allocator = allocator_type{})
      : m_id_map(allocator), m_allocator(allocator) {}

  /// \brief Destructor.
  ~compact_point_store() {
    if (m_data) {
      m_allocator.deallocate(m_data, m_num_points * m_num_dims);
      m_data = nullptr;
    }
  }

  // Copy constructor
  compact_point_store(const compact_point_store &other) = delete;

  // Move constructor
  compact_point_store(compact_point_store &&other) noexcept
      : m_num_points(other.m_num_points),
        m_num_dims(other.m_num_dims),
        m_data(other.m_data),
        m_id_map(std::move(other.m_id_map)),
        m_allocator(std::move(other.m_allocator)) {
    other.m_num_points = 0;
    other.m_num_dims   = 0;
    other.m_data       = nullptr;
  }

  // Copy assignment
  compact_point_store &operator=(const compact_point_store &other) = delete;

  // Move assignment
  compact_point_store &operator=(compact_point_store &&other) noexcept {
    if (this != &other) {
      if (m_data) {
        m_allocator.deallocate(m_data, m_num_points * m_num_dims);
        m_data = nullptr;
      }
      m_num_points = other.m_num_points;
      m_num_dims   = other.m_num_dims;
      m_data       = other.m_data;
      m_id_map     = std::move(other.m_id_map);
      m_allocator  = std::move(other.m_allocator);

      other.m_num_points = 0;
      other.m_num_dims   = 0;
      other.m_data       = nullptr;
    }
    return *this;
  }

  bool init(std::size_t num_points, std::size_t num_dims) {
    m_num_points = num_points;
    m_num_dims   = num_dims;

    m_id_map.clear();
    m_id_map.reserve(m_num_points);

    if (m_data) {
      std::cerr << "Data is already allocated." << std::endl;
      return false;
    }

    m_data = m_allocator.allocate(m_num_points * m_num_dims);
    return !!m_data;
  }

  inline bool contains(id_type id) const { return m_id_map.count(id) > 0; }

  inline std::size_t size() const { return num_points(); }

  inline bool empty() const { return size() == 0; }

  inline value_type *operator[](id_type id) {
    if (m_id_map.count(id) == 0) {
      const id_type internal_id = m_id_map.size();
      m_id_map[id]              = internal_id;
    }
    assert(m_id_map.count(id) > 0);
    const id_type internal_id = m_id_map.at(id);
    assert(internal_id < m_num_points);
    return data() + internal_id * m_num_dims;
  }

  inline const value_type *operator[](id_type id) const {
#ifndef NDEBUG
    if (m_id_map.count(id) == 0) {
      std::cerr << "ID " << id << " not found." << std::endl;
      std::abort();
    }
#endif
    assert(m_id_map.count(id) > 0);
    const id_type internal_id = m_id_map.at(id);
    assert(internal_id < m_num_points);
    return data() + internal_id * m_num_dims;
  }

  const value_type *at(id_type id) const { return (*this)[id]; }

  inline value_type *data() { return metall::to_raw_pointer(m_data); }

  inline const value_type *data() const { return metall::to_raw_pointer(m_data); }

  inline std::size_t num_points() const { return m_id_map.size(); }

  inline std::size_t dim() const { return m_num_dims; }

  const_iterator begin() const {
    return const_iterator(m_id_map.cbegin(), data());
  }

  const_iterator end() const { return const_iterator(m_id_map.cend(), data()); }

  const_id_iterator ids_begin() const {
    return const_id_iterator(m_id_map.cbegin());
  }

  const_id_iterator ids_begin() { return const_id_iterator(m_id_map.cbegin()); }

  const_id_iterator ids_end() const {
    return const_id_iterator(m_id_map.cend());
  }

  const_id_iterator ids_end() { return const_id_iterator(m_id_map.cend()); }

 private:
  std::size_t                                           m_num_points{0};
  std::size_t                                           m_num_dims{0};
  pointer                                               m_data{nullptr};
  id_table_t                                            m_id_map;
  dndetail::other_allocator<allocator_type, value_type> m_allocator;
};

template <typename _id_type, typename _value_type, typename _allocator_type>
class compact_point_store<_id_type, _value_type,
                          _allocator_type>::const_iterator {
 private:
  using id_table_iterator = typename id_table_t::const_iterator;

 public:
  using iterator_category = std::forward_iterator_tag;

  const_iterator() = default;

  explicit const_iterator(id_table_iterator itr, pointer data)
      : m_it(itr), m_data(data) {}

  const_iterator &operator++() {
    ++m_it;
    return *this;
  }

  const_iterator operator++(int) {
    const_iterator tmp(*this);
    ++m_it;
    return tmp;
  }

  bool operator==(const const_iterator &rhs) const { return m_it == rhs.m_it; }

  bool operator!=(const const_iterator &rhs) const { return m_it != rhs.m_it; }

  std::pair<id_type, value_type *> operator*() const {
    return priv_get_value();
  }

  std::pair<id_type, value_type *> operator->() const { priv_get_value(); }

 private:
  std::pair<id_type, value_type *> priv_get_value() const {
    const auto id          = m_it->first;
    const auto internal_id = m_it->second;
    return std::make_pair(id, m_data + internal_id);
  }

  id_table_iterator m_it;
  pointer           m_data;
};

template <typename _id_type, typename _value_type, typename _allocator_type>
class compact_point_store<_id_type, _value_type,
                          _allocator_type>::const_id_iterator {
 private:
  using internal_iterator = typename id_table_t::const_iterator;

 public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using pointer           = const id_type *;
  using reference         = const id_type &;

  const_id_iterator() = default;

  explicit const_id_iterator(internal_iterator itr) : m_it(itr) {}

  const_id_iterator &operator++() {
    ++m_it;
    return *this;
  }

  const_id_iterator operator++(int) {
    const_id_iterator tmp(*this);
    ++m_it;
    return tmp;
  }

  bool operator==(const const_id_iterator &rhs) const {
    return m_it == rhs.m_it;
  }

  bool operator!=(const const_id_iterator &rhs) const {
    return m_it != rhs.m_it;
  }

  reference operator*() const { return m_it->first; }

  pointer operator->() const { return &m_it->first; }

 private:
  internal_iterator m_it;
};

}  // namespace saltatlas