// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstdlib>
#include <memory>

#if __has_include(<metall/container/unordered_map.hpp>) \
  && __has_include(<metall/container/queue.hpp>)
#ifndef SALTATLAS_DNND_USE_METALL_CONTAINER
#define SALTATLAS_DNND_USE_METALL_CONTAINER 1
#endif
#endif

#if SALTATLAS_DNND_USE_METALL_CONTAINER
#include <metall/container/priority_queue.hpp>
#include <metall/container/unordered_map.hpp>
#else
#include <queue>
#include <unordered_map>
#endif

#if __has_include(<boost/unordered/unordered_flat_map.hpp>) \
&& __has_include(<boost/unordered/unordered_node_map.hpp>) \
&& defined(BOOST_VERSION) && BOOST_VERSION >= 108200
#ifndef SALTATLAS_DNND_USE_BOOST_OPEN_ADDRESS_CONTAINER
#define SALTATLAS_DNND_USE_BOOST_OPEN_ADDRESS_CONTAINER 1
#endif
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_node_map.hpp>
#endif

#include <saltatlas/dnnd/detail/utilities/allocator.hpp>
#include <saltatlas/dnnd/detail/utilities/float.hpp>

namespace saltatlas::dndetail {

namespace {
namespace container =
#if SALTATLAS_DNND_USE_METALL_CONTAINER
    metall::container;
#else
    std;
#endif
}  // namespace

template <typename Id, typename Distance>
struct neighbor {
  using id_type       = Id;
  using distance_type = Distance;

  neighbor() = default;

  neighbor(const id_type& _id, const distance_type& _distance)
      : id(_id), distance(_distance) {}

  friend bool operator<(const neighbor& lhd, const neighbor& rhd) {
    if (lhd.distance != rhd.distance) return lhd.distance < rhd.distance;
    return lhd.id < rhd.id;
  }

  template <typename T1, typename T2>
  friend std::ostream& operator<<(std::ostream& os, const neighbor<T1, T2>& ng);

  id_type       id;
  distance_type distance;
};

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const neighbor<T1, T2>& ng) {
  os << "id = " << ng.id << ", distance = " << ng.distance;
  return os;
}

template <typename Id, typename Distance>
inline bool operator==(const neighbor<Id, Distance>& lhs,
                       const neighbor<Id, Distance>& rhs) {
  return lhs.id == rhs.id && nearly_equal(lhs.distance, rhs.distance);
}

template <typename Id, typename Distance>
inline bool operator!=(const neighbor<Id, Distance>& lhs,
                       const neighbor<Id, Distance>& rhs) {
  return !(lhs == rhs);
}

// TODO: make a version that deos not take value?
template <typename Id, typename Distance, typename Value = std::byte,
          typename Alloc = std::allocator<std::byte>>
class unique_knn_heap {
 public:
  using id_type        = Id;
  using distance_type  = Distance;
  using value_type     = Value;
  using allocator_type = Alloc;
  using nenghbor_type  = neighbor<id_type, distance_type>;

 private:
  using heap_type = container::priority_queue<
      nenghbor_type,
      container::vector<nenghbor_type,
                        other_allocator<allocator_type, nenghbor_type>>>;

#if SALTATLAS_DNND_USE_BOOST_OPEN_ADDRESS_CONTAINER
  // Safeguard for not using offset-pointers with boost open address
  // TODO: check pointer type rather than allocator
  static_assert(
      std::is_same_v<allocator_type, std::allocator<std::byte>>,
      "Boost open address containers may not work with custom allocators");
  using map_type = boost::unordered_flat_map<
      id_type, value_type, std::hash<id_type>, std::equal_to<>,
      other_allocator<allocator_type, std::pair<const id_type, value_type>>>;
#else
  using map_type = container::unordered_map<
      id_type, value_type, std::hash<id_type>, std::equal_to<>,
      other_allocator<allocator_type, std::pair<const id_type, value_type>>>;
#endif

 public:
  explicit unique_knn_heap(const std::size_t k,
                           allocator_type    alloc = allocator_type{})
      : m_k(k), m_knn_heap(alloc), m_map(alloc) {
    m_map.reserve(k);
  }

  /// \brief Push a neighbor if it is closer than the current neighbors and is
  /// not one of the current neighbors.
  /// \param id Neighbor ID.
  /// \param d Distance.
  /// \param v Value associated with the neighbor.
  /// \return True if the neighbor has been pushed; otherwise, false.
  bool push_unique(const id_type& id, const distance_type& d,
                   value_type v = value_type{}) {
    if (m_map.count(id) > 0) return false;

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

  const nenghbor_type& top() { return m_knn_heap.top(); }

  const nenghbor_type& top() const { return m_knn_heap.top(); }

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