// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

// Memo:
// HIP memory does not work with RDMA.
#define SALTATLAS_SOLANET_APU_MATRIX_USE_HIP_MEMORY

#include <cstddef>
#include <cstdint>
#include <memory>

#include <hip/hip_runtime.h>
#include <spdlog/spdlog.h>

#include "saltatlas/solanet/detail/apu_nn/memory.hpp"
#include "saltatlas/solanet/detail/apu_nn/utils.hpp"

namespace saltatlas::solanet::apu_nn {

// Simple span class (like std::span)
template <typename T>
class span {
 public:
  SALTATLAS_HD_HD span() = default;

  SALTATLAS_HD_HD span(T* const data, const size_t size)
      : m_data(data), m_size(size) {}

  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE T& operator[](const size_t i) {
    return m_data[i];
  }

  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE const T& operator[](
      const size_t i) const {
    return m_data[i];
  }

  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE T*       data() { return m_data; }
  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE const T* data() const {
    return m_data;
  }

  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE size_t size() const {
    return m_size;
  }

  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE size_t mem_size() const {
    return m_size * sizeof(T);
  }

 private:
  T*     m_data = nullptr;
  size_t m_size = 0;
};

template <typename T>
class matrix_view {
 public:
  SALTATLAS_HD_HD matrix_view(T* const data, const size_t n_rows,
                              const size_t n_cols)
      : m_data(data), m_n_rows(n_rows), m_n_cols(n_cols) {}

  // Return pointer to the beginning of the specified row
  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE T* operator()(const size_t row) {
    return m_data + row * m_n_cols;
  }

  // Return const pointer to the beginning of the specified row
  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE const T* operator()(
      const size_t row) const {
    return m_data + row * m_n_cols;
  }

  // Return reference to the specified element
  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE T& operator()(const size_t row,
                                                         const size_t col) {
    return m_data[row * m_n_cols + col];
  }

  // Return const reference to the specified element
  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE const T& operator()(
      const size_t row, const size_t col) const {
    return m_data[row * m_n_cols + col];
  }

  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE T*       data() { return m_data; }
  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE const T* data() const {
    return m_data;
  }

  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE size_t n_rows() const {
    return m_n_rows;
  }
  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE size_t n_cols() const {
    return m_n_cols;
  }

  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE size_t size() const {
    return m_n_rows * m_n_cols;
  }

  SALTATLAS_HD_HD SALTATLAS_HD_FORCEINLINE size_t mem_size() const {
    return size() * sizeof(T);
  }

 private:
  T*     m_data   = nullptr;
  size_t m_n_rows = 0;
  size_t m_n_cols = 0;
};

// Simple matrix class for HIP device memory
template <typename T>
class matrix {
 public:
  matrix() = default;

  matrix(const size_t n_rows, const size_t n_cols) { reset(n_rows, n_cols); }

  // Copy constructor (deleted)
  matrix(const matrix<T>& other) = delete;
  // Copy assignment operator (deleted)
  matrix<T>& operator=(const matrix<T>& other) = delete;

  // Move constructor
  matrix(matrix<T>&& other) noexcept = default;
  // Move assignment operator
  matrix<T>& operator=(matrix<T>&& other) noexcept = default;

  ~matrix() { reset(); }

  // Return pointer to the beginning of the specified row
  T* operator()(const size_t row) { return m_data.get() + row * m_n_cols; }

  // Return const pointer to the beginning of the specified row
  const T* operator()(const size_t row) const {
    return m_data.get() + row * m_n_cols;
  }

  // Return reference to the specified element
  T& operator()(const size_t row, const size_t col) {
    return m_data.get()[row * m_n_cols + col];
  }

  // Return const reference to the specified element
  const T& operator()(const size_t row, const size_t col) const {
    return m_data.get()[row * m_n_cols + col];
  }

  // Reset the matrix
  void reset() {
    if (m_data != nullptr) {
      m_data.reset();
    }
    m_n_rows = 0;
    m_n_cols = 0;
  }

  // Reset the matrix with specified size
  bool reset(const size_t n_rows, const size_t n_cols) {
    reset();
#ifdef SALTATLAS_SOLANET_APU_MATRIX_USE_HIP_MEMORY
    m_data = make_hip_array<T>(n_rows * n_cols);
#else
    m_data = std::make_unique<T[]>(n_rows * n_cols);
#endif
    m_n_rows = n_rows;
    m_n_cols = n_cols;
    return m_data != nullptr;
  }

  bool is_initialized() const { return m_data != nullptr; }

  T*       data() { return m_data.get(); }
  const T* data() const { return m_data.get(); }

  size_t n_rows() const { return m_n_rows; }
  size_t n_cols() const { return m_n_cols; }

  matrix_view<T> get_view() {
    return matrix_view<T>(m_data.get(), m_n_rows, m_n_cols);
  }

  matrix_view<const T> get_const_view() const {
    return matrix_view<const T>(m_data.get(), m_n_rows, m_n_cols);
  }

  size_t size() const { return m_n_rows * m_n_cols; }

  size_t mem_size() const { return size() * sizeof(T); }

 private:
#ifdef SALTATLAS_SOLANET_APU_MATRIX_USE_HIP_MEMORY
  hip_unique_ptr<T> m_data{nullptr};
#else
  std::unique_ptr<T[]> m_data{nullptr};
#endif
  size_t m_n_rows = 0;
  size_t m_n_cols = 0;
};

}  // namespace saltatlas::solanet::apu_nn