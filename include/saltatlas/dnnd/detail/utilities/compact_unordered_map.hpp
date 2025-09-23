// Copyright 2020-2025 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

namespace saltatlas::detail::utilities {

/// \brief A simple and compact unordered map implementation that stores data in a contiguous array.
/// \tparam Key Key type.
/// \tparam T Mapped type.
/// \tparam CompactSizeType Compact size type. This type is used for the size and capacity of the map.
template <typename Key, typename T, typename CompactSizeType = uint16_t>
class compact_map {
 public:
  using key_type       = Key;
  using mapped_type    = T;
  using value_type     = std::pair<Key, T>;
  using const_iterator = const value_type*;
  using compact_size_type      = CompactSizeType;

  compact_map() = default;

  compact_map(const compact_map&)            = delete;
  compact_map& operator=(const compact_map&) = delete;
  compact_map(compact_map&&)                 = default;
  compact_map& operator=(compact_map&&)      = default;

  ~compact_map() = default;

  void reserve(const std::size_t capacity) {
    if (capacity >= std::numeric_limits<compact_size_type>::max()) {
      std::cerr << "The capacity of compact_map is too large" << std::endl;
      std::abort();
    }

    if (m_data) {
      if (capacity <= m_capacity) {
        return;
      }
      auto new_data = std::make_unique<value_type[]>(capacity);
      for (std::size_t i = 0; i < m_size; ++i) {
        new_data[i] = std::move(m_data[i]);
      }
      m_data = std::move(new_data);
    } else {
      m_data = std::make_unique<value_type[]>(capacity);
    }
    if (m_data == nullptr) {
      std::cerr << "Failed to allocate memory for compact_map" << std::endl;
      std::abort();
    }
    m_capacity = capacity;
  }

  std::size_t capacity() const { return m_capacity; }

  std::size_t size() const { return m_size; }

  std::size_t count(const key_type& key) const {
    for (std::size_t i = 0; i < m_size; ++i) {
      if (m_data[i].first == key) {
        return 1;
      }
    }
    return 0;
  }

  const mapped_type& at(const key_type& key) const {
    for (std::size_t i = 0; i < m_size; ++i) {
      if (m_data[i].first == key) {
        return m_data[i].second;
      }
    }
    std::cerr << "Key not found in compact_map " << key << std::endl;
    std::abort();
  }

  mapped_type& at(const key_type& key) {
    return const_cast<mapped_type&>(
        static_cast<const compact_map*>(this)->at(key));
  }

  const_iterator begin() const { return m_data.get(); }
  const_iterator end() const { return m_data.get() + m_size; }

  void emplace(key_type key, mapped_type value) {
    if (m_size == m_capacity) {
      std::cerr << "compact_map is full" << std::endl;
      std::abort();
    }
    m_data[m_size++] = std::make_pair(std::move(key), std::move(value));
  }

  void erase(const key_type& key) {
    for (std::size_t i = 0; i < m_size; ++i) {
      if (m_data[i].first == key) {
        if (m_capacity == 1) {
          m_size = 0;
        } else {
          m_data[i] = m_data[m_size - 1];
          --m_size;
        }
        return;
      }
    }
  }

 private:
  std::unique_ptr<value_type[]> m_data{nullptr};
  compact_size_type                      m_capacity{0};
  compact_size_type                      m_size{0};
};
}