// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>

#if __has_include(<metall/container/unordered_map.hpp>) \
  && __has_include(<metall/container/vector.hpp>)
#ifndef SALTATLAS_DNND_USE_METALL_CONTAINER
#define SALTATLAS_DNND_USE_METALL_CONTAINER 1
#endif
#endif

#if SALTATLAS_DNND_USE_METALL_CONTAINER
#include <metall/container/unordered_map.hpp>
#include <metall/container/vector.hpp>
#else
#include <unordered_map>
#include <vector>
#endif

#include <saltatlas/dnnd/detail/neighbor.hpp>
#include <saltatlas/dnnd/detail/utilities/allocator.hpp>

namespace saltatlas::dndetail {

namespace {
namespace container =
#if SALTATLAS_DNND_USE_METALL_CONTAINER
    metall::container;
#else
    std;
#endif
}  // namespace

template <typename IdType = uint64_t, typename DistanceType = double,
          typename Allocator = std::allocator<std::byte>>
class nn_index {
 public:
  using id_type        = IdType;
  using distance_type  = DistanceType;
  using neighbor_type  = neighbor<id_type, distance_type>;
  using allocator_type = Allocator;

 private:
  using neighbor_list_type =
      container::vector<neighbor_type,
                        other_allocator<allocator_type, neighbor_type>>;
  using point_table_type = container::unordered_map<
      id_type, neighbor_list_type, std::hash<id_type>, std::equal_to<>,
      other_scoped_allocator<allocator_type,
                             std::pair<const id_type, neighbor_list_type>>>;

  using point_iterator          = typename point_table_type::iterator;
  using const_point_iterator    = typename point_table_type::const_iterator;
  using neighbor_iterator       = typename neighbor_list_type::iterator;
  using const_neighbor_iterator = typename neighbor_list_type::const_iterator;

 public:
  explicit nn_index(const allocator_type &allocator = allocator_type{})
      : m_index(allocator) {}

  void insert(const id_type &source, const neighbor_type &neighbor) {
    m_index[source].push_back(neighbor);
  }

  void sort_neighbors(const id_type &source) {
    std::sort(m_index[source].begin(), m_index[source].end());
  }

  void sort_and_remove_duplicate_neighbors(const id_type &source) {
    sort_neighbors(source);
    m_index[source].erase(
        std::unique(m_index[source].begin(), m_index[source].end()),
        m_index[source].end());
  }

  /// \warning The neighbor list must be sorted beforehand.
  void prune_neighbors(const id_type    &source,
                       const std::size_t num_max_neighbors) {
    if (m_index.at(source).size() > num_max_neighbors) {
      m_index[source].resize(num_max_neighbors);
    }
  }

  auto points_begin() { return m_index.begin(); }

  auto points_end() { return m_index.end(); }

  auto points_begin() const { return m_index.begin(); }

  auto points_end() const { return m_index.end(); }

  auto neighbors_begin(const id_type &source) {
    return m_index[source].begin();
  }

  auto neighbors_end(const id_type &source) { return m_index[source].end(); }

  auto neighbors_begin(const id_type &source) const {
    assert(m_index.count(source) > 0);
    return m_index.at(source).begin();
  }

  auto neighbors_end(const id_type &source) const {
    assert(m_index.count(source) > 0);
    return m_index.at(source).end();
  }

  std::size_t num_points() const { return m_index.size(); }

  std::size_t num_neighbors(const id_type &source) const {
    if (m_index.count(source) == 0) return 0;
    return m_index.at(source).size();
  }

  void clear() { m_index.clear(); }

  void clear_neighbors(const id_type &source) { m_index[source].clear(); }

  allocator_type get_allocator() const { return m_index.get_allocator(); }

  bool empty() const { return m_index.empty(); }

 private:
  point_table_type m_index;
};

}  // namespace saltatlas::dndetail