// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstdlib>
#include <memory>

#include <queue>

#include <boost/container/vector.hpp>
#include <boost/version.hpp>
#if defined(BOOST_VERSION) && BOOST_VERSION >= 108700
#include <boost/unordered/unordered_flat_map.hpp>
#else
#error "Boost 1.87.00 or higher is required."
#endif

#include <saltatlas/common/detail/neighbor.hpp>
#include <saltatlas/common/detail/utilities/float.hpp>
#include <saltatlas/dnnd/detail/utilities/allocator.hpp>

#if SALTATLAS_DNND_KNN_HEAP_USE_COMPACT_MAP
#include <saltatlas/dnnd/detail/utilities/compact_unordered_map.hpp>
#endif

namespace saltatlas::dndetail {

#if SALTATLAS_DNND_KNN_HEAP_USE_COMPACT_MAP
#error "Compact map is not supported for now."
#endif

/// \brief Data structure to store up to k nearest neighbors without duplicate
/// neighbor IDs. Each neighbor can have an associated value in addition to
/// distance.
template <typename Id, typename Distance, typename Value = std::byte,
          typename Alloc = std::allocator<std::byte>>
class unique_knn_heap {
 public:
  using id_type        = Id;
  using distance_type  = Distance;
  using value_type     = Value;
  using allocator_type = Alloc;
  using neighbor_type  = detail::neighbor<id_type, distance_type>;

 private:
  using heap_type = std::priority_queue<
      neighbor_type,
      boost::container::vector<neighbor_type,
                               other_allocator<allocator_type, neighbor_type>>>;

  using map_type = boost::unordered_flat_map<
      id_type, value_type, std::hash<id_type>, std::equal_to<>,
      other_allocator<allocator_type, std::pair<const id_type, value_type>>>;

 public:
  explicit unique_knn_heap(const std::size_t k,
                           allocator_type    alloc = allocator_type{})
      : m_k(k), m_knn_heap(alloc), m_map(alloc) {
    m_map.reserve(k);
  }

  /// \brief Push a neighbor if it is closer than the current farthest neighbor
  /// and is not one of the current neighbors.
  /// \param id Neighbor ID.
  /// \param d Distance.
  /// \param v Value associated with the neighbor.
  /// \return True if the neighbor has been pushed; otherwise, false.
  bool try_add(const id_type& id, const distance_type& d,
               value_type v = value_type{}) {
    if (m_map.count(id) > 0) {
      return false;
    }

    if (m_knn_heap.size() < m_k) {
      priv_push_nocheck(id, d, v);
      return true;
    }

    if (m_knn_heap.top().distance > d) {
      pop();
      priv_push_nocheck(id, d, v);
      return true;
    }

    return false;
  }

  const neighbor_type& top() { return m_knn_heap.top(); }

  const neighbor_type& top() const { return m_knn_heap.top(); }

  void pop() {
    assert(m_map.count(m_knn_heap.top().id) > 0);
    m_map.erase(m_knn_heap.top().id);
    m_knn_heap.pop();
  }

  bool contains(const id_type& id) const { return m_map.count(id); }

  // Provide only const iterators to prevent the user from modifying the IDs.
  typename map_type::const_iterator begin() const { return m_map.begin(); }

  // Provide only const iterators to prevent the user from modifying the IDs.
  typename map_type::const_iterator end() const { return m_map.end(); }

  value_type& value(const id_type& id) { return m_map.at(id); }

  const value_type& value(const id_type& id) const { return m_map.at(id); }

  std::size_t size() const { return m_knn_heap.size(); }

  bool empty() const { return m_knn_heap.empty(); }

  std::size_t k() const { return m_k; }

  /// \brief Return neighbors.
  std::vector<neighbor_type> extract_neighbors() const {
    std::vector<neighbor_type> neighbors;
    neighbors.reserve(m_knn_heap.size());
    auto tmp = m_knn_heap;
    while (!tmp.empty()) {
      neighbors.emplace_back(tmp.top());
      tmp.pop();
    }
    std::reverse(neighbors.begin(), neighbors.end());
    return neighbors;
  }

 private:
  void priv_push_nocheck(const id_type& id, const distance_type& d,
                         value_type v) {
    m_knn_heap.emplace(id, d);
    m_map.emplace(id, std::move(v));
  }

  std::size_t m_k;
  heap_type   m_knn_heap;
  map_type    m_map;
};

}  // namespace saltatlas::dndetail
