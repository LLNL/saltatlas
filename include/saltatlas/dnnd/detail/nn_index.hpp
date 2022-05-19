// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <unordered_map>
#include <vector>

#include <saltatlas/dnnd/detail/neighbor.hpp>
#include <saltatlas/dnnd/detail/utilities/allocator.hpp>

namespace saltatlas::dndetail {

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
      std::vector<neighbor_type,
                  other_allocator<allocator_type, neighbor_type>>;
  using point_table_type = std::unordered_map<
      id_type, neighbor_list_type, std::hash<id_type>, std::equal_to<>,
      other_scoped_allocator<allocator_type,
                             std::pair<const id_type, neighbor_list_type>>>;

  using point_iterator          = typename point_table_type::iterator;
  using const_point_iterator    = typename point_table_type::const_iterator;
  using neighbor_iterator       = typename neighbor_list_type::iterator;
  using const_neighbor_iterator = typename neighbor_list_type::const_iterator;

 public:
  void insert(const id_type &source, const neighbor_type& neighbor) {
    m_index[source].push_back(neighbor);
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

 private:
  point_table_type m_index;
};

}  // namespace saltatlas::dndetail