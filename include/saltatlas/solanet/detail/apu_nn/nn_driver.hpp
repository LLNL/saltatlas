// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#define SALTATLAS_SOLANET_APU_NND_OPTIMIZE_QUERY_GRAPH

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <hip/hip_runtime.h>
#include <spdlog/spdlog.h>

#include "saltatlas/dnnd/utility.hpp"
#include "saltatlas/solanet/detail/apu_nn/memory.hpp"
#include "saltatlas/solanet/detail/apu_nn/nndescent.hpp"
#include "saltatlas/solanet/detail/apu_nn/search_knng_builder.hpp"
#include "saltatlas/solanet/detail/cuvs_nn/common.hpp"
#include "saltatlas/solanet/detail/nn_index_view.hpp"
#include "saltatlas/solanet/detail/point_store_view.hpp"

namespace saltatlas::solanet::apu_nn {

namespace {
namespace d3cvs                  = saltatlas::solanet::cuvs_nn;
constexpr int k_merge_block_size = 128;

template <typename T>
SALTATLAS_HD_GLOBAL void add_scalar_kernel(T* data, const size_t n,
                                           const T value) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] += value;
  }
}

/// Updates one `master` row per thread by merging candidate neighbors.
template <typename IDType, typename DistType>
SALTATLAS_HD_GLOBAL void update_index_kernel(
    const IDType* candidate_ids, const DistType* candidate_dists,
    const size_t candidate_rows, const size_t candidate_cols,
    const IDType candidate_id_offset, IDType* master_ids,
    DistType* master_dists, const size_t master_rows, const size_t master_cols,
    const IDType master_id_offset, const bool negate_candidate_dists) {
  const size_t sid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (sid >= master_rows || master_cols == 0) {
    return;
  }

  IDType* const master_ids_row = master_ids + sid * master_cols;
  if (master_id_offset != IDType{0}) {
    for (size_t i = 0; i < master_cols; ++i) {
      master_ids_row[i] += master_id_offset;
    }
  }

  if (sid >= candidate_rows || candidate_cols == 0) {
    return;
  }

  const IDType* const candidate_ids_row = candidate_ids + sid * candidate_cols;
  DistType*           master_dists_row  = master_dists + sid * master_cols;
  const DistType* candidate_dists_row = candidate_dists + sid * candidate_cols;
  const int       tail                = static_cast<int>(master_cols - 1);
  for (size_t c = 0; c < candidate_cols; ++c) {
    DistType cd = candidate_dists_row[c];
    if (negate_candidate_dists) {
      cd = -cd;
    }
    if (!(cd < master_dists_row[tail])) {
      continue;
    }
    const IDType cid = candidate_ids_row[c] + candidate_id_offset;
    int          pos = tail;
    while (pos > 0 && cd < master_dists_row[pos - 1]) {
      master_dists_row[pos] = master_dists_row[pos - 1];
      master_ids_row[pos]   = master_ids_row[pos - 1];
      --pos;
    }
    master_dists_row[pos] = cd;
    master_ids_row[pos]   = cid;
  }
}
}  // namespace

template <typename IDType, typename FeatureElemType, typename DistanceType>
class driver {
 public:
  using id_type   = IDType;
  using dist_type = DistanceType;
  using fe_type   = FeatureElemType;

  using point_store_view_type = d3dtl::point_store_view<id_type, fe_type>;
  using point_type            = typename point_store_view_type::point_type;
  using const_point_type = typename point_store_view_type::const_point_type;
  using point_store_type = matrix<fe_type>;
  using index_view_type  = d3dtl::nn_index_view<IDType, DistanceType>;
  using index_nid_type   = matrix<id_type>;
  using index_dist_type  = matrix<dist_type>;

  struct index_type {
    index_nid_type  nids;
    index_dist_type dists;
  };

 private:
  // CAGRA Index: dataset and knng
  using cagra_index_t = cuvs::neighbors::cagra::index<fe_type, id_type>;
  using rmm_mem_pool_t =
      rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;

 public:
  explicit driver(const bool verbose = false) : m_verbose(verbose) {}

  // Disable for now..
  driver(const driver&)             = delete;
  driver& operator=(const driver&)  = delete;
  driver(const driver&&)            = delete;
  driver& operator=(const driver&&) = delete;

  ~driver() { reset(); }

  // Allocate memory for receiving the point store and index data from other
  // ranks
  auto alloc_recv_buf(const size_t size_bytes) {
    return make_hip_array<unsigned char>(size_bytes);
  }

  index_view_type make_index_view(index_type& index) const {
    return index_view_type(index.nids.n_rows(), index.nids.n_cols(),
                           index.nids.data(), index.dists.data());
  }

  index_view_type make_index_view(const index_type& index) const {
    return index_view_type(index.nids.n_rows(), index.nids.n_cols(),
                           const_cast<id_type*>(index.nids.data()),
                           const_cast<dist_type*>(index.dists.data()));
  }

  index_type build_index_object(const point_store_type& pstore,
                                const std::string_view  dist_func,
                                const size_t k, const size_t nnd_k,
                                const double rho, const double delta) {
    if (dist_func != "l2" && dist_func != "inner_product" &&
        dist_func != "ip") {
      throw std::invalid_argument(
          "Only l2 and inner_product distances are supported.");
    }

    const auto n_points = pstore.n_rows();
    if (n_points == 0 || k == 0) {
      return index_type{};
    }
    if (!pstore.data()) {
      throw std::invalid_argument("Point store is not initialized.");
    }
    if (nnd_k < k) {
      throw std::invalid_argument("nnd_k must be >= k.");
    }

    spdlog::trace("Building base KNNG");
    rec_time().start("Build-base-knng");
    auto nnd_knng = apu_nn::build_index<id_type, fe_type, dist_type>(
        matrix_view<fe_type>(const_cast<fe_type*>(pstore.data()),
                             pstore.n_rows(), pstore.n_cols()),
        dist_func, nnd_k, static_cast<float>(rho), static_cast<float>(delta));
    auto local_nids  = std::move(nnd_knng.first);
    auto local_dists = std::move(nnd_knng.second);
    if (nnd_k > k) {
      // Keep only the top-k per row. Direct contiguous slicing would corrupt
      // row boundaries when nnd_k > k.
      matrix<id_type>   topk_nids(n_points, k);
      matrix<dist_type> topk_dists(n_points, k);
      SALTATLAS_HIP_CHECK(
          hipMemcpy2D(topk_nids.data(), k * sizeof(id_type), local_nids.data(),
                      nnd_k * sizeof(id_type), k * sizeof(id_type), n_points,
                      hipMemcpyDeviceToDevice));
      SALTATLAS_HIP_CHECK(hipMemcpy2D(
          topk_dists.data(), k * sizeof(dist_type), local_dists.data(),
          nnd_k * sizeof(dist_type), k * sizeof(dist_type), n_points,
          hipMemcpyDeviceToDevice));
      local_nids  = std::move(topk_nids);
      local_dists = std::move(topk_dists);
    }
    rec_time().stop();  // Build-base-knng

    index_type out;
    out.nids.reset(local_nids.n_rows(), local_nids.n_cols());
    out.dists.reset(local_dists.n_rows(), local_dists.n_cols());
    const size_t n_elements = out.nids.size();
    SALTATLAS_HIP_CHECK(hipMemcpy(out.nids.data(), local_nids.data(),
                                  n_elements * sizeof(id_type),
                                  hipMemcpyDeviceToDevice));
    SALTATLAS_HIP_CHECK(hipMemcpy(out.dists.data(), local_dists.data(),
                                  n_elements * sizeof(dist_type),
                                  hipMemcpyDeviceToDevice));
    SALTATLAS_HIP_CHECK(hipDeviceSynchronize());

    // Prepare for CAGRA search
    if (dist_func == "l2") {
      m_cagra_dist_func = cuvs::distance::DistanceType::L2Expanded;
    } else if (dist_func == "inner_product" || dist_func == "ip") {
      m_cagra_dist_func = cuvs::distance::DistanceType::InnerProduct;
    } else if (dist_func == "cosine") {
      m_cagra_dist_func = cuvs::distance::DistanceType::CosineExpanded;
    } else {
      std::cerr << "Unsupported distance: " << dist_func << std::endl;
      std::abort();
    }

    return out;
  }

  index_nid_type replicate(const index_nid_type& nids) {
    index_nid_type out;
    out.reset(nids.n_rows(), nids.n_cols());
    const size_t n_elements = nids.size();
    SALTATLAS_HIP_CHECK(hipMemcpy(out.data(), nids.data(),
                                  n_elements * sizeof(id_type),
                                  hipMemcpyDeviceToDevice));
    SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
    return out;
  }

  void add_id_offset(const id_type offset, index_nid_type& inex_nids) {
    if (offset > 0) {
      const size_t  n_elements   = inex_nids.size();
      constexpr int k_elem_block = 256;
      const dim3    elem_block(k_elem_block);
      const dim3    elem_grid((n_elements + k_elem_block - 1) / k_elem_block);
      hipLaunchKernelGGL((add_scalar_kernel<id_type>), elem_grid, elem_block, 0,
                         nullptr, inex_nids.data(), n_elements, offset);
      SALTATLAS_HIP_CHECK(hipGetLastError());
    }
    SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
  }

  // Allocate point store
  // The ownership of the allocated point store is passed to the caller.
  point_store_type alloc_point_store(const size_t n_rows, const size_t dims) {
    point_store_type pstore;
    if (n_rows == 0 || dims == 0) {
      return pstore;
    }
    pstore.reset(n_rows, dims);
    return pstore;
  }

  void optimize_index(const index_view_type& index) {
    if (index.size() == 0 || index.num_neighbors() == 0) {
      return;
    }

    if (index.neighbor_ids_data() == nullptr) {
      throw std::invalid_argument(
          "optimize_index requires valid neighbor ID and distance buffers.");
    }

    auto knng_ids =
        matrix_view<id_type>(const_cast<id_type*>(index.neighbor_ids_data()),
                             index.size(), index.num_neighbors());
    auto knng_dists =
        (index.dists_data())
            ? matrix_view<dist_type>(const_cast<dist_type*>(index.dists_data()),
                                     index.size(), index.num_neighbors())
            : matrix_view<dist_type>(nullptr, 0, 0);
    make_optimized_query_graph_apu<id_type, dist_type>(
        knng_ids, knng_dists, knng_ids,
        (index.dists_data()) ? query_graph_distance_mode::actual_distance
                             : query_graph_distance_mode::knng_position);
  }

  void update_index(const index_view_type& candidates,
                    const id_type          candidate_id_offset,
                    const id_type master_id_offset, index_type& master_index) {
    const size_t n_rows = master_index.nids.n_rows();
    const size_t k      = master_index.nids.n_cols();
    if (n_rows == 0 || k == 0) {
      return;
    }
    if (candidates.size() == 0 || candidates.num_neighbors() == 0) {
      return;
    }
    if (master_index.nids.data() == nullptr ||
        candidates.neighbor_ids_data() == nullptr) {
      throw std::invalid_argument(
          "update_index requires valid candidate/master neighbor ID buffers.");
    }

    const dim3 block(k_merge_block_size);
    const dim3 grid((n_rows + k_merge_block_size - 1) / k_merge_block_size);
    // CAGRA postprocess emits InnerProduct scores in max-close form (dot).
    // This pipeline stores min-close distances (-dot), so negate candidates.
    const bool negate_candidate_dists =
        (m_cagra_dist_func == cuvs::distance::DistanceType::InnerProduct);
    hipLaunchKernelGGL((update_index_kernel<id_type, dist_type>), grid, block,
                       0, nullptr, candidates.neighbor_ids_data(),
                       candidates.dists_data(), candidates.size(),
                       candidates.num_neighbors(), candidate_id_offset,
                       master_index.nids.data(), master_index.dists.data(),
                       n_rows, k, master_id_offset, negate_candidate_dists);
    SALTATLAS_HIP_CHECK(hipGetLastError());
    SALTATLAS_HIP_CHECK(hipDeviceSynchronize());
  }

  index_view_type run_queries(const point_store_view_type& query_pstore,
                              const point_store_view_type& trg_pstore,
                              const index_view_type&       trg_index,
                              const bool                   copy_query_to_gpu,
                              const int frontier_size, const int query_k) {
    if (query_k <= 0) {
      throw std::invalid_argument("query_k must be > 0.");
    }
    if (query_pstore.size() == 0 || query_pstore.dims() == 0) {
      return index_view_type();
    }

    priv_run_query_cagra(query_pstore, trg_pstore, trg_index, copy_query_to_gpu,
                         frontier_size, query_k);
    return index_view_type(
        m_query_result_nids.extent(0), m_query_result_nids.extent(1),
        m_query_result_nids.data_handle(), m_query_result_dists.data_handle());
  }

  void reset() {}

 private:
  void priv_run_query_cagra(const point_store_view_type& query_pstore,
                            const point_store_view_type& trg_pstore,
                            const index_view_type&       trg_index,
                            const bool                   copy_query_to_gpu,
                            const int frontier_size, const int query_k) {
    auto queries = d3cvs::make_dev_matrix_view(
        query_pstore.data(), query_pstore.size(), query_pstore.dims());
    const auto n_queries = queries.extent(0);

    d3cvs::d_matrix_type<fe_type>            local_trg_pstore{m_dev_res};
    d3cvs::d_matrix_type<id_type>            local_trg_index{m_dev_res};
    d3cvs::d_matrix_view_type<const fe_type> local_trg_pstore_view;
    d3cvs::d_matrix_view_type<const id_type> local_trg_index_view;
    if (copy_query_to_gpu) {
      spdlog::trace("Copy query data");
      saltatlas::rec_time().start("Copy-query-data");
      local_trg_pstore = d3cvs::copy_to_dev(
          d3cvs::make_host_matrix_view(trg_pstore.data(), trg_pstore.size(),
                                       trg_pstore.dims()),
          m_dev_res);
      local_trg_index =
          d3cvs::copy_to_dev(d3cvs::make_host_matrix_view(
                                 trg_index.neighbor_ids_data(),
                                 trg_index.size(), trg_index.num_neighbors()),
                             m_dev_res);
      saltatlas::rec_time().stop();  // Copy-query-data
      local_trg_pstore_view = d3cvs::make_const_matrix_view(local_trg_pstore);
      local_trg_index_view  = d3cvs::make_const_matrix_view(local_trg_index);
    } else {
      local_trg_pstore_view = d3cvs::make_dev_matrix_view(
          trg_pstore.data(), trg_pstore.size(), trg_pstore.dims());
      local_trg_index_view = d3cvs::make_dev_matrix_view(
          trg_index.neighbor_ids_data(), trg_index.size(),
          trg_index.num_neighbors());
    }

    if (m_query_result_nids.extent(0) != n_queries ||
        m_query_result_nids.extent(1) != static_cast<size_t>(query_k)) {
      priv_setup_rmm(n_queries, query_k);
      m_query_result_nids =
          d3cvs::make_dev_matrix<id_type>(n_queries, query_k, m_dev_res);
      m_query_result_dists =
          d3cvs::make_dev_matrix<dist_type>(n_queries, query_k, m_dev_res);
    }

    // https://github.com/ROCm-DS/hipVS/blob/release/rocmds-25.10/cpp/include/cuvs/neighbors/cagra.hpp#L175
    cuvs::neighbors::cagra::search_params search_params;
    const int                             effective_frontier_size =
        (frontier_size > 0) ? frontier_size : (query_k * 2);
    search_params.itopk_size           = effective_frontier_size;
    search_params.search_width         = 1;
    search_params.num_random_samplings = 1;
    search_params.team_size            = 8;
    search_params.algo = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
    search_params.max_iterations = trg_index.size();
    // search_params.min_iterations = search_params.itopk_size * 2;

    // CAGRA Index: dataset and knng
    spdlog::trace("Const CAGRA index");
    saltatlas::rec_time().start("Const-cagra-index");
    using cagra_index_t = cuvs::neighbors::cagra::index<fe_type, id_type>;
    cagra_index_t cagra_index =
        priv_const_cagra_index(local_trg_pstore_view, local_trg_index_view);
    raft::resource::sync_stream(m_dev_res);
    saltatlas::rec_time().stop();

    // spdlog::trace("Alloc query result buffers");
    // auto result_nids = d3cvs::make_dev_matrix<id_type>(n_queries, k,
    // m_dev_res); auto result_dists =
    //     d3cvs::make_dev_matrix<dist_type>(n_queries, k, m_dev_res);

    spdlog::trace("Search");
    saltatlas::rec_time().start("Search");
    cuvs::neighbors::cagra::search(m_dev_res, search_params, cagra_index,
                                   queries, m_query_result_nids.view(),
                                   m_query_result_dists.view());
    raft::resource::sync_stream(m_dev_res);
    const auto elapsed_sec = saltatlas::rec_time().stop();
    spdlog::trace("Finished searching");
    // Show search performance in QPS (Queries Per Second)
    const double qps = n_queries / elapsed_sec;
    spdlog::trace(
        "Search QPS: #of queries {} / Elapsed time: {:.3f} sec = {} QPS",
        n_queries, elapsed_sec, saltatlas::utility::add_comma_separators(qps));
    const double throughput = trg_index.size() * n_queries / elapsed_sec;
    spdlog::trace(
        "(Search throughput: #of queries: {} X KNNG size: {}) / Elapsed time: "
        "{:.3f} sec = Throughput: {}",
        n_queries, trg_index.size(), elapsed_sec,
        saltatlas::utility::add_comma_separators(throughput));

    // return std::make_pair(std::move(result_nids), std::move(result_dists));
  }

  cagra_index_t priv_const_cagra_index(
      d3cvs::d_matrix_view_type<const fe_type> dataset,
      d3cvs::d_matrix_view_type<const id_type> knng) {
    try {
      // std::cout << "CAGRA index dataset = " << dataset.extent(0) << " x "
      //           << dataset.extent(1) << std::endl;
      // std::cout << "CAGRA index graph = " << knng.extent(0) << " x "
      //           << knng.extent(1) << std::endl;
      return cagra_index_t(m_dev_res, m_cagra_dist_func, dataset, knng);
    } catch (std::bad_alloc& e) {
      std::cerr << "Insufficient GPU memory to construct CAGRA index with "
                   "dataset on GPU"
                << std::endl;
      std::abort();
    } catch (raft::logic_error& e) {
      std::cerr << "Insufficient GPU memory to construct CAGRA index with "
                   "dataset on GPU"
                << std::endl;
      std::abort();
    }

    // Dummy return value to avoid compiler warning
    return cagra_index_t(m_dev_res);
  }

  void priv_setup_rmm(const size_t n_queries, const int query_k) {
    // result nids and dists for query search.
    size_t cuvs_memory_pool_reserve_bytes =
        n_queries * query_k * sizeof(id_type) +
        n_queries * query_k * sizeof(dist_type);
    cuvs_memory_pool_reserve_bytes *= 2;
    // Align to 256MB for better memory pool performance
    constexpr size_t k_align = 256ULL << 20;
    cuvs_memory_pool_reserve_bytes =
        ((cuvs_memory_pool_reserve_bytes + k_align - 1) / k_align) * k_align;
    // Set a minimum pool size of 1GB to avoid too small pool size when the
    // dataset is small
    cuvs_memory_pool_reserve_bytes =
        std::max<size_t>(cuvs_memory_pool_reserve_bytes, 1ULL << 30);

    if (m_verbose) {
      std::cout << "hipVS memory pool reserve : "
                << (float)cuvs_memory_pool_reserve_bytes / (1ULL << 30) << " GB"
                << std::endl;
    }

    // This value is not correct. However, since there is only GPU per rank,
    // this should be fine for now.
    const int n_node_local_ranks = 0;
    if (m_verbose) {
      std::cout << gpu::get_cuvs_gpu_info(n_node_local_ranks) << std::endl;
    }

    m_rmm_pool = std::make_unique<rmm_mem_pool_t>(
        rmm::mr::get_current_device_resource(), cuvs_memory_pool_reserve_bytes);
  }

  raft::device_resources          m_dev_res;
  std::unique_ptr<rmm_mem_pool_t> m_rmm_pool;
  cuvs::distance::DistanceType    m_cagra_dist_func;
  bool                            m_verbose{false};

  d3cvs::d_matrix_type<id_type>   m_query_result_nids{m_dev_res};
  d3cvs::d_matrix_type<dist_type> m_query_result_dists{m_dev_res};
};

}  // namespace saltatlas::solanet::apu_nn
