// Copyright 2022–2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <memory>
#include <type_traits>

#if __has_include(<metall/container/unordered_map.hpp>)
#ifndef SALTATLAS_USE_METALL_CONTAINER
#define SALTATLAS_USE_METALL_CONTAINER 1
#endif
#endif

#if SALTATLAS_USE_METALL_CONTAINER
#include <metall/container/unordered_map.hpp>
#else
#include <unordered_map>
#endif

#include "saltatlas/dnnd/detail/utilities/allocator.hpp"

namespace saltatlas {

namespace {
namespace container =
#if SALTATLAS_USE_METALL_CONTAINER
    metall::container;
#else
    std;
#endif
}  // namespace

/// \brief A container to store points.
/// The container is designed to store points with unique IDs.
/// The container is implemented using an unordered map container.
/// \tparam ID Point ID type.
/// \tparam PointType Point type.
/// \tparam Hasher Hash function for ID.
/// \tparam EqualTo Equal function for ID.
/// \tparam Allocator Allocator type.
template <typename ID, typename PointType, typename Hasher = std::hash<ID>,
          typename EqualTo   = std::equal_to<ID>,
          typename Allocator = std::allocator<std::byte>>
class point_store {
 public:
  using id_type        = ID;
  using point_type     = PointType;
  using hasher         = Hasher;
  using equal_to       = EqualTo;
  using allocator_type = Allocator;

 private:
  using point_table_type = container::unordered_map<
      id_type, PointType, hasher, equal_to,
      dndetail::other_scoped_allocator<allocator_type,
                                       std::pair<const id_type, PointType>>>;

 public:
  using iterator       = typename point_table_type::iterator;
  using const_iterator = typename point_table_type::const_iterator;

 public:
  /// \brief Constructor.
  explicit point_store(const allocator_type &allocator = allocator_type{})
      : m_points_table(allocator) {}

  /// \brief Constructor.
  explicit point_store(std::size_t bucket_count, const hasher &hash = hasher(),
                       const equal_to  &equal = equal_to(),
                       const Allocator &alloc = Allocator())
      : m_points_table(bucket_count, hash, equal, alloc) {}

  /// \brief Copy constructor.
  point_store(const point_store &other) = default;

  /// \brief Move constructor.
  point_store(point_store &&other) noexcept = default;

  /// \brief Destructor.
  ~point_store() noexcept = default;

  /// \brief Copy assignment operator.
  point_store &operator=(const point_store &other) = default;

  /// \brief Move assignment operator.
  point_store &operator=(point_store &&other) noexcept = default;

  // \brief Allocator aware copy constructor.
  point_store(const point_store &other, const allocator_type &allocator)
      : m_points_table(other.m_points_table, allocator) {}

  // \brief Allocator aware move constructor.
  point_store(point_store &&other, const allocator_type &allocator) noexcept
      : m_points_table(std::move(other.m_points_table), allocator) {}

  /// \brief Checks if a point is contained.
  /// \param id Point ID.
  /// \return Returns true if point with 'id' is contained; otherwise, returns
  /// false.
  bool contains(const id_type &id) const { return m_points_table.count(id); }

  /// \brief Returns the number of points contained.
  /// \return The number of points contained.
  std::size_t size() const { return m_points_table.size(); }

  /// \brief Returns true if the container is empty.
  /// \return True if the container is empty; otherwise, false.
  bool empty() const { return m_points_table.empty(); }

  /// \brief Returns the point data associated with 'id'.
  /// A new instance is allocated if the corresponding item does not exist.
  /// \param id Point ID.
  /// \return The point data associated with 'id'.
  point_type &operator[](const id_type &id) { return m_points_table[id]; }

  /// \brief Returns the point data associated with 'id'. Const version.
  /// \param id Point ID.
  /// \return The point data associated with 'id'.
  const point_type &operator[](const id_type &id) const {
    assert(m_points_table.count(id) > 0);
    return m_points_table.at(id);
  }

  /// \brief Returns the point data associated with 'id'.
  /// \param  id Point ID.
  /// \return The point data associated with 'id'.
  const point_type &at(const id_type &id) const {
    assert(m_points_table.count(id) > 0);
    return m_points_table.at(id);
  }

  /// \brief Erase the point data associated with 'id'.
  /// \param id Point ID to erase.
  void erase(const id_type &id) { m_points_table.erase(id); }

  /// \brief Returns an iterator that points the first element.
  /// \return An iterator that points the first element.
  iterator begin() { return m_points_table.begin(); }

  /// \brief Returns an iterator that points the next element of the last one.
  /// \return An iterator that points the next element of the last one.
  iterator end() { return m_points_table.end(); }

  /// \brief Returns an iterator that points the first element. Const version.
  /// \return An iterator that points the first element.
  const_iterator begin() const { return m_points_table.begin(); }

  /// \brief Returns an iterator that points the next element of the last one.
  /// Const version.
  /// \return An iterator that points the next element of the last one.
  const_iterator end() const { return m_points_table.end(); }

  /// \brief Clears all contents and reduce the storage usage.
  void reset() {
    m_points_table.clear();
    m_points_table.rehash(0);
  }

  /// \brief Reserves internal storage to hold 'n' points.
  /// \param n Reservation size.
  void reserve(const std::size_t n) { m_points_table.reserve(n); }

 private:
  point_table_type m_points_table;
};

}  // namespace saltatlas