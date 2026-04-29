// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <type_traits>

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#endif

#if __has_include(<cuvs/neighbors/cagra.hpp>)
#include <cuvs/neighbors/cagra.hpp>
#endif

#if __has_include(<cuvs/neighbors/cagra_graph.hppp>)
#include <cuvs/neighbors/cagra_graph.hppp>
#endif

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "saltatlas/solanet/detail/cuvs_nn/utils.hpp"

namespace saltatlas::solanet::cuvs_nn {
using rmm_mem_pool_type =
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;

using raft_index_t = int64_t;
template <typename T>
using d_matrix_type = raft::device_matrix<T, raft_index_t>;

template <typename T>
using d_matrix_view_type = raft::device_matrix_view<T, raft_index_t>;

template <typename T>
using h_matrix_type = raft::pinned_matrix<T, raft_index_t>;

template <typename T>
using h_matrix_view_type = raft::host_matrix_view<T, raft_index_t>;

template <typename T>
inline h_matrix_type<T> make_host_matrix(const size_t     n_rows,
                                         const size_t     n_cols,
                                         raft::resources& host_res) {
  return raft::make_pinned_matrix<T, raft_index_t>(host_res, n_rows, n_cols);
}

template <typename T>
inline d_matrix_type<T> make_dev_matrix(const size_t            n_rows,
                                        const size_t            n_cols,
                                        raft::device_resources& dev_res) {
  return raft::make_device_matrix<T, raft_index_t>(dev_res, n_rows, n_cols);
}

template <typename MatrixT>
inline h_matrix_view_type<typename MatrixT::element_type> make_host_matrix_view(
    MatrixT& matrix) {
  return raft::make_host_matrix_view<typename MatrixT::element_type,
                                     raft_index_t>(
      matrix.data_handle(), matrix.extent(0), matrix.extent(1));
}

template <typename MatrixT>
inline d_matrix_view_type<typename MatrixT::element_type> make_dev_matrix_view(
    MatrixT& matrix) {
  return raft::make_device_matrix_view<typename MatrixT::element_type,
                                       raft_index_t>(
      matrix.data_handle(), matrix.extent(0), matrix.extent(1));
}

template <typename T>
inline h_matrix_view_type<T> make_host_matrix_view(T* const     data,
                                                   const size_t n_rows,
                                                   const size_t n_cols) {
  return raft::make_host_matrix_view<T, raft_index_t>(data, n_rows, n_cols);
}

template <typename T>
inline d_matrix_view_type<T> make_dev_matrix_view(T* const     data,
                                                  const size_t n_rows,
                                                  const size_t n_cols) {
  return raft::make_device_matrix_view<T, raft_index_t>(data, n_rows, n_cols);
}

template <typename T>
inline auto make_const_matrix_view(const h_matrix_type<T>& matrix) {
  return raft::make_const_mdspan(matrix.view());
}

template <typename T>
inline auto make_const_matrix_view(const d_matrix_type<T>& matrix) {
  return raft::make_const_mdspan(matrix.view());
}

template <typename T>
inline auto make_const_matrix_view(const h_matrix_view_type<T>& matrix_view) {
  return raft::make_const_mdspan(matrix_view);
}

template <typename T>
inline auto make_const_matrix_view(const d_matrix_view_type<T>& matrix_view) {
  return raft::make_const_mdspan(matrix_view);
}

// Free matrix
// Do not invoke destructor so that
// A) calling this function multiple times to the same matrix is safe
// B) checking matrix.size() == 0 is possible
template <typename T>
inline void free_matrix(h_matrix_type<T>& matrix, raft::resources& host_res) {
  matrix = make_host_matrix<T>(0, 0, host_res);
}

template <typename T>
inline void free_matrix(d_matrix_type<T>&       matrix,
                        raft::device_resources& dev_res) {
  matrix = make_dev_matrix<T>(0, 0, dev_res);
}

template <typename MatrixViewT>
auto copy_to_host(MatrixViewT matrix_view, raft::resources& host_res,
                  raft::device_resources& dev_res) {
  using T       = std::remove_cv_t<typename MatrixViewT::element_type>;
  auto h_matrix = make_host_matrix<T>(matrix_view.extent(0),
                                      matrix_view.extent(1), host_res);

  auto stream = raft::resource::get_cuda_stream(dev_res);
  raft::copy(h_matrix.data_handle(), matrix_view.data_handle(),
             matrix_view.size(), stream);
  raft::resource::sync_stream(dev_res, stream);
  return h_matrix;
}

template <typename MatrixViewT>
inline auto copy_to_dev(MatrixViewT             h_matrix_view,
                        raft::device_resources& dev_res) {
  using T       = std::remove_cv_t<typename MatrixViewT::element_type>;
  auto d_matrix = make_dev_matrix<T>(h_matrix_view.extent(0),
                                     h_matrix_view.extent(1), dev_res);

  auto stream = raft::resource::get_cuda_stream(dev_res);
  raft::copy(d_matrix.data_handle(), h_matrix_view.data_handle(),
             h_matrix_view.size(), stream);
  raft::resource::sync_stream(dev_res, stream);

  std::cout << "Copy to device done." << std::endl;

  return d_matrix;
}

/// Allocate and copy matrix to device
/// colms_to_skip: #of columns to skip copying.
template <typename MatrixViewT>
inline auto strided_copy_to_dev(MatrixViewT             matrix_view,
                                const size_t            n_colms_to_copy,
                                raft::device_resources& dev_res) {
  using T = std::remove_const_t<typename MatrixViewT::element_type>;
  if (n_colms_to_copy > size_t(matrix_view.extent(1))) {
    std::cerr << "Too large to copy " << n_colms_to_copy << " size"
              << std::endl;
    std::abort();
  }
  auto d_matrix =
      make_dev_matrix<T>(matrix_view.extent(0), n_colms_to_copy, dev_res);

  gpu::copy_to_device_2d(d_matrix.data_handle(), n_colms_to_copy,
                         matrix_view.data_handle(), matrix_view.extent(1),
                         matrix_view.extent(0));

  return d_matrix;
}
}  // namespace saltatlas::solanet::cuvs_nn
