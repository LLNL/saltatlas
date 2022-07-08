// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <memory>

#if __has_include(<metall/container/unordered_map.hpp>)
#ifndef SALTATLAS_DNND_USE_METALL_CONTAINER
#define SALTATLAS_DNND_USE_METALL_CONTAINER 1
#endif
#endif

#if SALTATLAS_DNND_USE_METALL_CONTAINER
#include <metall/container/unordered_map.hpp>
#else
#include <unordered_map>
#endif

#include <saltatlas/dnnd/detail/feature_vector.hpp>
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

template <typename ID, typename FeatureElement,
          typename Allocator = std::allocator<std::byte>>
class point_store {
 public:
  using id_type              = ID;
  using feature_element_type = FeatureElement;
  using allocator_type       = Allocator;
  using feature_vector_type =
      dndetail::feature_vector<FeatureElement, Allocator>;

 private:
  using point_table_type = container::unordered_map<
      id_type, feature_vector_type, std::hash<id_type>, std::equal_to<>,
      other_scoped_allocator<allocator_type,
                             std::pair<const id_type, feature_vector_type>>>;

 public:
  explicit point_store(const allocator_type& allocator = allocator_type{})
      : m_point_table(allocator) {}

  bool contains(const id_type &id) const { return m_point_table.count(id); }

  std::size_t size() const { return m_point_table.size(); }

  feature_vector_type &feature_vector(const id_type &id) {
    m_max_id = std::max(id, m_max_id);
    return m_point_table[id];
  }

  const feature_vector_type &feature_vector(const id_type &id) const {
    assert(m_point_table.count(id) > 0);
    return m_point_table.at(id);
  }

  auto begin() { return m_point_table.begin(); }

  auto end() { return m_point_table.end(); }

  auto begin() const { return m_point_table.begin(); }

  auto end() const { return m_point_table.end(); }

  void clear() { m_point_table.clear(); }

  void reserve(const std::size_t n) { m_point_table.reserve(n); }

  id_type max_id() const { return m_max_id; }

 private:
  point_table_type m_point_table;
  id_type          m_max_id{0};
};

}  // namespace saltatlas::dndetail