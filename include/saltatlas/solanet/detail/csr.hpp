// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <scoped_allocator>
#include <string>
#include <vector>

#include <metall/offset_ptr.hpp>

#include "saltatlas/dnnd/detail/utilities/omp.hpp"

namespace saltatlas::solanet {
template <typename ValueType>
class csr {
 public:
  using index_type  = size_t;
  using value_type  = ValueType;
  using offset_type = std::size_t;

  class range_proxy {
   public:
    range_proxy(const value_type *begin, const value_type *end)
        : m_begin(begin), m_end(end) {}

    const value_type *begin() const { return m_begin; }
    const value_type *end() const { return m_end; }

   private:
    const value_type *m_begin{nullptr};
    const value_type *m_end{nullptr};
  };

  csr() = default;

  csr(const std::size_t num_indices, const std::size_t num_values) {
    init(num_indices, num_values);
  }

  // Copy constructor
  csr(const csr &other) = delete;

  // Copy assignment
  csr &operator=(const csr &other) = delete;

  // Move constructor
  csr(csr &&other) noexcept
      : m_buf(std::move(other.m_buf)),
        m_offsets(other.m_offsets),
        m_values(other.m_values),
        m_num_indices(other.m_num_indices),
        m_num_values(other.m_num_values) {
    other.m_offsets     = nullptr;
    other.m_values      = nullptr;
    other.m_num_indices = 0;
    other.m_num_values  = 0;
  }

  // Move assignment
  csr &operator=(csr &&other) noexcept {
    if (this != &other) {
      m_buf         = std::move(other.m_buf);
      m_offsets     = other.m_offsets;
      m_values      = other.m_values;
      m_num_indices = other.m_num_indices;
      m_num_values  = other.m_num_values;

      other.m_offsets     = nullptr;
      other.m_values      = nullptr;
      other.m_num_indices = 0;
      other.m_num_values  = 0;
    }
    return *this;
  }

  void *get_buffer() const { return m_buf.get(); }

  size_t buffer_size() const {
    const std::size_t bytes = (m_num_indices + 1) * sizeof(offset_type) +
                              m_num_values * sizeof(index_type);
    return bytes;
  }

  offset_type *offsets() const { return m_offsets; }
  offset_type *offsets() { return m_offsets; }

  value_type *values(const std::size_t index) const {
    return m_values + m_offsets[index];
  }

  value_type *values(const std::size_t index) {
    return m_values + m_offsets[index];
  }

  std::size_t num_indices() const { return m_num_indices; }
  std::size_t num_values() const { return m_num_values; }
  std::size_t num_values(const std::size_t index) const {
    return m_offsets[index + 1] - m_offsets[index];
  }

  range_proxy operator[](const std::size_t index) const {
    return range_proxy(m_values + m_offsets[index],
                       m_values + m_offsets[index + 1]);
  }

  range_proxy operator[](const std::size_t index) {
    return range_proxy(m_values + m_offsets[index],
                       m_values + m_offsets[index + 1]);
  }

  range_proxy at(const std::size_t index) const {
    return this->operator[](index);
  }

  range_proxy at(const std::size_t index) { return this->operator[](index); }

  template <typename iterator>
  void build_index(iterator count_begin, iterator count_end) {
    const std::size_t num_indices = std::distance(count_begin, count_end);
    size_t            num_values  = 0;
    for (auto it = count_begin; it != count_end; ++it) {
      num_values += *it;
    }

    priv_allocate_arrays(num_indices, num_values);

    // Fill in the offsets
    m_offsets[0] = 0;
    for (std::size_t i = 0; i < num_indices; ++i) {
      m_offsets[i + 1] = m_offsets[i] + *(count_begin + i);
    }
    assert(m_offsets[num_indices] == num_values);
  }

  void print(std::ostream &os) const {
    for (std::size_t i = 0; i < m_num_indices; ++i) {
      os << i << ": ";
      for (const auto &v : this->operator[](i)) {
        os << v << " ";
      }
      os << std::endl;
    }
  }

  void clear() {
    m_buf.reset();
    m_offsets     = nullptr;
    m_values      = nullptr;
    m_num_indices = 0;
    m_num_values  = 0;
  }

  void init(const std::size_t num_indices, const std::size_t num_values) {
    priv_allocate_arrays(num_indices, num_values);
  }

 private:
  void priv_allocate_arrays(const std::size_t num_indices,
                            const std::size_t num_values) {
    assert(!m_buf);
    m_num_indices = num_indices;
    m_num_values  = num_values;

    m_buf = std::make_unique<std::byte[]>(buffer_size());

    m_offsets = reinterpret_cast<offset_type *>(m_buf.get());
    m_values  = reinterpret_cast<value_type *>(
        m_buf.get() +
        std::ptrdiff_t((m_num_indices + 1) * sizeof(offset_type)));
  }

  std::unique_ptr<std::byte[]> m_buf{nullptr};
  offset_type                 *m_offsets{nullptr};
  value_type                  *m_values{nullptr};
  std::size_t                  m_num_indices{0};
  std::size_t                  m_num_values{0};
};

}  // namespace saltatlas::solanet