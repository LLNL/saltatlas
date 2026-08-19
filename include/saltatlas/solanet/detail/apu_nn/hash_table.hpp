// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <type_traits>
#include <utility>

#if !defined(__CUDACC__)
#include <hip/hip_runtime.h>
#endif

#include "saltatlas/solanet/detail/apu_nn/utils.hpp"

namespace saltatlas::solanet::apu_nn {

// Simple and compact hash table
// Takes an array from the caller side, there is not internal memory allocation
// Designed for small capacity.
template <typename KeyType>
class simple_set_kernel {
 public:
  simple_set_kernel() = default;

  ~simple_set_kernel() = default;

  SALTATLAS_HD_HD static void init(const int      capacity,
                                   const KeyType  invalid_key,
                                   KeyType* const keys) {
    // Initialize keys
    for (int i = 0; i < capacity; ++i) {
      keys[i] = invalid_key;
    }
  }

  // Returns true if the key was added, false if the key already exists
  SALTATLAS_HD_HD static std::pair<int, bool> add(const KeyType  key,
                                                  const int      capacity,
                                                  const KeyType  invalid_key,
                                                  KeyType* const keys) {
    const int hash = static_cast<int>(key) % capacity;
    for (int i = 0; i < static_cast<int>(capacity); ++i) {
      const int idx = (hash + i) % capacity;
      if (keys[idx] == invalid_key) {
        // Empty slot found, add the key
        keys[idx] = key;
        return {idx, true};
      } else if (keys[idx] == key) {
        // Key found
        return {idx, false};
      }
    }
    assert(false);
    return {-1, false};  // Table full, key not found
  }

  SALTATLAS_HD_HD static bool contains(const KeyType key, const int capacity,
                                       const KeyType  invalid_key,
                                       KeyType* const keys) {
    const int hash = static_cast<int>(key) % capacity;
    for (int i = 0; i < static_cast<int>(capacity); ++i) {
      const int idx = (hash + i) % capacity;
      if (keys[idx] == invalid_key) {
        // Empty slot found, key not present
        return false;
      } else if (keys[idx] == key) {
        // Key found
        return true;
      }
    }
    return false;  // Table full, key not found
  }

  SALTATLAS_HD_HD static int find(const KeyType key, const int capacity,
                                  const KeyType  invalid_key,
                                  KeyType* const keys) {
    const int hash = static_cast<int>(key) % capacity;
    for (int i = 0; i < static_cast<int>(capacity); ++i) {
      const int idx = (hash + i) % capacity;
      if (keys[idx] == invalid_key) {
        // Empty slot found, key not present
        return -1;
      } else if (keys[idx] == key) {
        // Key found
        return idx;
      }
    }
    return -1;  // Table full, key not found
  }
};

// Use simple_set_kernel internally
// Wrapper class for easier usage
template <typename KeyType, KeyType k_invalid_key = static_cast<KeyType>(-1)>
class simple_set {
 public:
  SALTATLAS_HD_HD simple_set() = default;

  SALTATLAS_HD_HD simple_set(const int capacity, KeyType* const bufs)
      : m_capacity(capacity), m_bufs(bufs) {}

  ~simple_set() = default;

  SALTATLAS_HD_HD int capacity() const { return m_capacity; }

  SALTATLAS_HD_HD void clear() {
    simple_set_kernel<KeyType>::init(m_capacity, k_invalid_key, m_bufs);
  }

  SALTATLAS_HD_HD bool add(const KeyType key) {
    return simple_set_kernel<KeyType>::add(key, m_capacity, k_invalid_key,
                                           m_bufs)
        .second;
  }

  SALTATLAS_HD_HD bool contains(const KeyType key) {
    return simple_set_kernel<KeyType>::contains(key, m_capacity, k_invalid_key,
                                                m_bufs);
  }

 private:
  int      m_capacity = 0;
  KeyType* m_bufs     = nullptr;
};

// simple map implementation using simple_set_kernel
template <typename KeyType, typename ValueType,
          KeyType k_invalid_key = static_cast<KeyType>(-1)>
class simple_map {
 public:
  SALTATLAS_HD_HD simple_map() = default;

  SALTATLAS_HD_HD simple_map(const int capacity, KeyType* const key_bufs,
                             ValueType* const value_bufs)
      : m_capacity(capacity), m_key_bufs(key_bufs), m_value_bufs(value_bufs) {}

  ~simple_map() = default;

  SALTATLAS_HD_HD int capacity() const { return m_capacity; }

  SALTATLAS_HD_HD void clear() {
    simple_set_kernel<KeyType>::init(m_capacity, k_invalid_key, m_key_bufs);
  }

  SALTATLAS_HD_HD bool insert(const KeyType key, const ValueType value) {
    const auto [idx, inserted] = simple_set_kernel<KeyType>::add(
        key, m_capacity, k_invalid_key, m_key_bufs);
    if (inserted) {
      m_value_bufs[idx] = value;
    }
    return inserted;
  }

  SALTATLAS_HD_HD bool contains(const KeyType key) {
    return simple_set_kernel<KeyType>::contains(key, m_capacity, k_invalid_key,
                                                m_key_bufs);
  }

  SALTATLAS_HD_HD ValueType get(const KeyType   key,
                                const ValueType default_value) {
    const int idx = simple_set_kernel<KeyType>::find(key, m_capacity,
                                                     k_invalid_key, m_key_bufs);
    if (idx != -1) {
      return m_value_bufs[idx];
    } else {
      return default_value;
    }
  }

 private:
  int        m_capacity   = 0;
  KeyType*   m_key_bufs   = nullptr;
  ValueType* m_value_bufs = nullptr;
};

}  // namespace saltatlas::solanet::apu_nn