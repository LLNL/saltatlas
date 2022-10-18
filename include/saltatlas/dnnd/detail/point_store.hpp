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
  using iterator       = typename point_table_type::iterator;
  using const_iterator = typename point_table_type::const_iterator;

 public:

  /// \brief Constructor.
  /// \param allocator Allocator instance.
  explicit point_store(const allocator_type &allocator = allocator_type{})
      : m_point_table(allocator) {}

  /// \brief Checks if a point is contained.
  /// \param id Point ID.
  /// \return Returns true if point with 'id' is contained; otherwise, returns
  /// false.
  bool contains(const id_type &id) const { return m_point_table.count(id); }

  /// \brief Returns the number of points contained.
  /// \return The number of points contained.
  std::size_t size() const { return m_point_table.size(); }

  /// \brief Returns the feature vector of point 'id'.
  /// A new instance is allocated if the corresponding item does not exist.
  /// \param id Point ID.
  /// \return The feature vector of point 'id'.
  feature_vector_type &feature_vector(const id_type &id) {
    m_max_id = std::max(id, m_max_id);
    return m_point_table[id];
  }

  /// \brief Returns the feature vector of point 'id'. Const version.
  /// \param id Point ID.
  /// \return The feature vector of point 'id'.
  const feature_vector_type &feature_vector(const id_type &id) const {
    assert(m_point_table.count(id) > 0);
    return m_point_table.at(id);
  }

  /// \brief Sets the feature of point 'id' to
  /// the elements from range ['first', 'last').
  /// \tparam iterator Iterator type of an input feature.
  /// \param id Point ID.
  /// \param first Iterator pointing to the first element inserted.
  /// \param last Iterator pointing to the end element inserted.
  template <typename iterator>
  void set(const id_type &id, iterator first, iterator last) {
    auto& f = feature_vector(id);
    f.clear();
    f.insert(f.begin(), first, last);
  }

  /// \brief Returns an iterator that points the first element.
  /// \return An iterator that points the first element.
  iterator begin() { return m_point_table.begin(); }

  /// \brief Returns an iterator that points the next element of the last one.
  /// \return An iterator that points the next element of the last one.
  iterator end() { return m_point_table.end(); }

  /// \brief Returns an iterator that points the first element. Const version.
  /// \return An iterator that points the first element.
  const_iterator begin() const { return m_point_table.begin(); }

  /// \brief Returns an iterator that points the next element of the last one.
  /// Const version.
  /// \return An iterator that points the next element of the last one.
  const_iterator end() const { return m_point_table.end(); }

  /// \brief Clear contents.
  void clear() { m_point_table.clear(); }

  /// \brief Reserve internal storage to hold 'n' elements.
  /// \param n Reservation size.
  void reserve(const std::size_t n) { m_point_table.reserve(n); }

  /// \brief Returns the max point ID.
  /// \return Max point ID.
  id_type max_id() const { return m_max_id; }

 private:
  point_table_type m_point_table;
  id_type          m_max_id{0};
};

}  // namespace saltatlas::dndetail