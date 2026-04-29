// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <boost/unordered/unordered_flat_map.hpp>
#include <metall/detail/memory.hpp>
#include <metall/utility/hash.hpp>

#include "saltatlas/common/detail/neighbor.hpp"
#include "saltatlas/common/detail/utilities/general.hpp"
#include "saltatlas/dnnd/detail/utilities/omp.hpp"
#include "saltatlas/dnnd/utility.hpp"
#include "saltatlas/neo_dnnd/mpi.hpp"
#include "saltatlas/solanet/detail/nn_index_view.hpp"
#include "saltatlas/solanet/detail/point_reader.hpp"
#include "saltatlas/solanet/detail/point_store_view.hpp"
#include "saltatlas/solanet/detail/utilities/mutex.hpp"
#include "saltatlas/solanet/singleton_time_recorder.hpp"

#ifdef SALTATLAS_SOLANET_CUVS
#include "saltatlas/solanet/detail/cuvs_nn/nn_driver.hpp"
#endif

#ifdef SALTATLAS_SOLANET_OMP
#include "saltatlas/solanet/detail/omp_nn/nn_driver.hpp"
#endif

#include "saltatlas/solanet/detail/apu_nn/nn_driver.hpp"

#define SALTATLAS_SOLANET_DOUBLE_BUFFERING 1

namespace saltatlas::solanet {
namespace {
namespace omp   = saltatlas::utility::omp;
namespace mutex = saltatlas::dndetail::mutex;
namespace bst   = boost;
namespace bstuo = boost::unordered;
}  // namespace

template <typename id_type, typename dist_type, typename knng_type,
          typename partitioner_type>
void optimize_knng(knng_type& knng, const partitioner_type& partitioner,
                   mpi::communicator& comm,
                   const size_t max_degree = std::numeric_limits<size_t>::max(),
                   const bool   verbose    = false);

// Partition points over MPI ranks
template <typename id_type>
static auto get_partitioner(const int mpi_size) {
  auto partitioner = [=](const id_type id) -> int {
    return metall::utility::hash<id_type, 6378>{}(id) % mpi_size;
  };
  return partitioner;
};

/// \brief Forward declaration of the main SOLANET engine class
template <typename IDType, typename FeatureElemType, typename DistanceType,
          template <typename T, typename F, typename D> class NNDriver>
class basic_solanet_engine;

#ifdef SALTATLAS_SOLANET_OMP
/// \brief SOLANET engine class using OpenMP backend.
template <typename IDType, typename FeatureElemType, typename DistanceType>
using solanet_engine_omp =
    basic_solanet_engine<IDType, FeatureElemType, DistanceType, omp_nn::driver>;
#endif

#ifdef SALTATLAS_SOLANET_CUVS
/// \brief SOLANET engine class using CUVS backend.
template <typename IDType, typename FeatureElemType, typename DistanceType>
using solanet_engine_cuvs = basic_solanet_engine<IDType, FeatureElemType,
                                                 DistanceType, cuvs_nn::driver>;
#endif

/// \brief SOLANET engine class using APU backend.
template <typename IDType, typename FeatureElemType, typename DistanceType>
using solanet_engine_apu =
    basic_solanet_engine<IDType, FeatureElemType, DistanceType, apu_nn::driver>;

/// \brief Basic SOLANET engine class implementation.
template <typename IDType, typename FeatureElemType, typename DistanceType,
          template <typename T, typename F, typename D> class NNDriver>
class basic_solanet_engine {
 public:
  using id_type       = IDType;
  using dist_type     = DistanceType;
  using fe_type       = FeatureElemType;
  using neighbor_type = detail::neighbor<id_type, dist_type>;
  using sparse_knng_type =
      bstuo::unordered_flat_map<id_type, std::vector<neighbor_type>>;

  struct options {
    size_t      k{0};
    size_t      nnd_k{0};  // If 0, fall back to k.
    double      delta{0.0001};
    double      rho{0.5};
    std::string dist_func{"l2"};  // "l2" or cosine for cuvs
    // Number of groups to split the neighbors into for the final
    // refinement step. If <= 1, no splitting is done.
    int refine_final_n_groups{4};
    int query_k{0};  // Number of neighbors to search for in the query phase. If
                     // 0, it will be set to k.
    int  frontier_size{0};  // Search frontier size. If 0, use query_k * 2.
    bool verbose{
        false};  // Legacy fallback when constructor verbose is omitted.
  };

 private:
  // point store used by the data reader
  using reader_point_store_type = point_store<id_type, std::vector<fe_type>>;

  using nn_driver          = NNDriver<id_type, fe_type, dist_type>;
  using pstore_object_type = typename nn_driver::point_store_type;
  using index_view_type    = typename nn_driver::index_view_type;
  using pstore_view_type   = typename nn_driver::point_store_view_type;
  using index_object_type  = typename nn_driver::index_type;
  using index_nid_type     = typename nn_driver::index_nid_type;

  // Todo: check which code requires id_type to be uint32_t
  // static_assert(std::is_same_v<id_type, uint32_t>, "id_type must be
  // uint32_t");

 public:
  explicit basic_solanet_engine(const options& opt, mpi::communicator& comm)
      : basic_solanet_engine(opt, comm, opt.verbose) {}

  basic_solanet_engine(const options& opt, mpi::communicator& comm,
                       const bool verbose)
      : m_opt(opt), m_comm(comm), m_nn_driver(verbose) {
    m_opt.verbose = verbose;
#ifdef SALTATLAS_SOLANET_CUVS
    // Show a warning if k is not a multiple of 32
    if (m_opt.k % 32 != 0) {
      spdlog::warn(
          "k is not a multiple of 32. This may cause an issue in the "
          "nn_descent class in cuVS.");
    }
#endif
    if (m_opt.nnd_k == 0) {
      m_opt.nnd_k = m_opt.k;
    }
    priv_show_dram_usage(__LINE__);
  }

  sparse_knng_type build_index(
      const std::vector<std::filesystem::path>& pstore_paths,
      const std::string&                        pstore_format) {
    {
      spdlog::info("Read points");
      rec_time().start("read-points");

      auto org_pstore = dn3detail::read_points<id_type, fe_type>(
          pstore_paths, pstore_format, get_partitioner<id_type>(m_comm.size()),
          m_opt.verbose, m_comm);

      m_g_n_points = m_comm.all_reduce_sum(org_pstore.size());
      rec_time().stop();

      m_dims = m_comm.all_reduce_max(
          org_pstore.size() > 0 ? org_pstore.begin()->second.size() : 0);

      spdlog::trace("Construct local point store");
      rec_time().start("Const-pstore");
      priv_const_local_pstore(org_pstore);
      rec_time().stop();
    }

    // This timer is stopped at the end of this funciton
    rec_time().start("SOLANET-build-main");

    spdlog::info("Build local index");
    rec_time().start("Build-local-index");
    auto local_index_obj =
        m_nn_driver.build_index_object(m_pstore_obj, m_opt.dist_func, m_opt.k,
                                       m_opt.nnd_k, m_opt.rho, m_opt.delta);
    rec_time().stop();  // End of Build-local-knng

    rec_time().start("(barrier-wait-all-local-knng-const)");
    m_comm.barrier();
    rec_time().stop();

    index_object_type master_index_obj;

    if (m_comm.size() > 1) {
      rec_time().start("Refine-indices");
      priv_refine_knngs(std::move(local_index_obj), master_index_obj);
      rec_time().stop();
    } else {
      master_index_obj = std::move(local_index_obj);
    }

    spdlog::trace("Make global o2e id maps");
    rec_time().start("Make-global-o2e-id-maps");
    const auto global_o2e_id_map = priv_make_global_o2e_id_maps();
    rec_time().stop();

    m_comm.barrier();

    spdlog::trace("Make final kNNG");
    rec_time().start("Make-final-knng");
    // do this on the driver side?
    auto final_knng =
        priv_conv_to_final_graph(global_o2e_id_map, master_index_obj);
    rec_time().stop();

    m_comm.barrier();
    rec_time().stop();  // End of SOLANET-build-main

    rec_time().start("Clean-up");
    m_pstore_obj.reset();
    master_index_obj.nids.reset();
    master_index_obj.dists.reset();
    m_split_sizes.clear();
    m_split_sizes.shrink_to_fit();
    m_pid_offsets.clear();
    m_pid_offsets.shrink_to_fit();
    m_l2e_id_table.clear();
    m_l2e_id_table.shrink_to_fit();
    m_nn_driver.reset();
    rec_time().stop();  // End of Clean-up

    priv_show_dram_usage(__LINE__);

    return final_knng;
  }

  /// \brief Optimize kNNG. Specifically, make the graph undirected and prune
  /// high-degree vertices..
  void optimize(sparse_knng_type& knng,
                const size_t max_degree = std::numeric_limits<size_t>::max()) {
    optimize_knng<id_type, dist_type>(knng,
                                      get_partitioner<id_type>(m_comm.size()),
                                      m_comm, max_degree, m_opt.verbose);
  }

 private:
  void priv_const_local_pstore(const reader_point_store_type& org_pstore) {
    // Share split sizes among all ranks
    m_comm.all_gather(org_pstore.size(), m_split_sizes);
    for (size_t i = 0; i < m_split_sizes.size(); ++i) {
      if (m_opt.verbose && m_comm.rank() == 0) {
        std::cout << "Partition " << i << ", size: " << m_split_sizes[i]
                  << std::endl;
      }
    }

    // Calculate global offsets for each split
    m_pid_offsets.resize(m_split_sizes.size(), 0);
    for (size_t i = 0; i < m_split_sizes.size() - 1; ++i) {
      m_pid_offsets[i + 1] = m_pid_offsets[i] + m_split_sizes[i];
    }

    if (org_pstore.empty()) {
      return;
    }

    // Compute local to external ID table
    spdlog::trace("Create local to external ID table");
    m_l2e_id_table.resize(org_pstore.size());
    auto pitr = org_pstore.begin();
    for (size_t i = 0; i < org_pstore.size(); ++i, ++pitr) {
      assert(pitr != org_pstore.end());
      const auto eid    = pitr->first;
      m_l2e_id_table[i] = eid;
    }

    // Copy the points to local point stores
    const auto dims = org_pstore.begin()->second.size();
    // Allocate point store in the device memory.
    spdlog::trace("Allocate memory for point store");
    m_pstore_obj = m_nn_driver.alloc_point_store(org_pstore.size(), dims);
    auto local_pstore =
        pstore_view_type(org_pstore.size(), dims, m_pstore_obj.data());
    spdlog::trace("Copy points to point store");
    OMP_DIRECTIVE(parallel) {
      const auto [begin, end] = detail::partial_range(
          org_pstore.size(), omp::get_thread_num(), omp::get_num_threads());
      auto pitr = org_pstore.begin();
      std::advance(pitr, begin);
      for (size_t lpi = begin; lpi < end; ++lpi, ++pitr) {
        if (pitr->second.size() != dims) {
          m_comm.cerr() << "Different feature vector dimension size, expected "
                        << dims << ", but got " << pitr->second.size()
                        << " with ID " << pitr->first << std::endl;
          m_comm.abort();
        }
        const auto& point = pitr->second;
        std::memcpy(local_pstore[lpi].data(), point.data(),
                    point.size() * sizeof(fe_type));
      }
    }  // OMP parallel over points within a split
  }

  void priv_refine_knngs(index_object_type&& local_index_obj,
                         index_object_type&  master_index_obj) {
    const int requested_final_n_groups = (m_opt.refine_final_n_groups > 0)
                                             ? m_opt.refine_final_n_groups
                                             : m_comm.size();
    const int refine_final_n_groups =
        std::max(2, std::min(requested_final_n_groups, m_comm.size()));

    // Current merge/refine implementation assumes power-of-two groups.
    if ((m_comm.size() & (m_comm.size() - 1)) != 0 ||
        (refine_final_n_groups & (refine_final_n_groups - 1)) != 0) {
      m_comm.cerr()
          << "Only power-of-two MPI size and refine_final_n_groups are "
             "supported"
          << std::endl;
      m_comm.abort();
    }

    spdlog::trace("Merge indices, {} to {} groups", m_comm.size(),
                  refine_final_n_groups);

    spdlog::trace("Create MPI Wins");
    rec_time().start("Create-MPI-Wins");
    auto pstore_comm = priv_create_mpi_win_for_pstore();
    auto index_comm  = priv_create_mpi_win_for_index(
        local_index_obj.nids.data(),
        local_index_obj.nids.size() * sizeof(id_type));
    m_comm.barrier();
    rec_time().stop();

    spdlog::info("Start binary-tree-based kNNG refinement");
    rec_time().start("Binary-merge-tree-refinement");
    priv_refine_knngs_binary_merge_tree(local_index_obj, pstore_comm,
                                        index_comm, refine_final_n_groups);
    rec_time().stop();

    priv_show_dram_usage(__LINE__, false);

    spdlog::info("Start group-based flat kNNG refinement");
    index_nid_type group_index_nids;
    rec_time().start("Merge-and-optimize-grouped-knng");
    priv_merge_and_optimize_group_knng(local_index_obj, group_index_nids,
                                       refine_final_n_groups, index_comm);
    m_comm.barrier();
    rec_time().stop();

    {
      spdlog::trace("Generate master index");
      const int group_size       = m_comm.size() / refine_final_n_groups;
      const int group_no         = m_comm.rank() / group_size;
      const int group_begin_rank = group_no * group_size;
      // master index's IDs are offseted to the global ID space
      rec_time().start("Gen-master-index");
      master_index_obj = std::move(local_index_obj);
      m_nn_driver.add_id_offset(m_pid_offsets[group_begin_rank],
                                master_index_obj.nids);
      rec_time().stop();
    }

    // Recreate MPI win for the merged index for the final refinement step.
    rec_time().start("Setup-rdma-group-index");
    index_comm.free();
    auto group_index_comm = priv_create_mpi_win_for_index(
        group_index_nids.data(), group_index_nids.size() * sizeof(id_type));
    rec_time().stop();

    // Next, do the same search-based refinement with each merged partioon
    // group without merge.
    spdlog::info("Start flat kNNG refinement with #of groups {}",
                 refine_final_n_groups);
    rec_time().start("Flat-knng-refinement");
    priv_refine_knngs_flat(master_index_obj, pstore_comm, group_index_comm,
                           refine_final_n_groups);
    rec_time().stop();

    group_index_comm.free();
    pstore_comm.free();
  }

  void priv_refine_knngs_binary_merge_tree(index_object_type& local_index_obj,
                                           mpi::rdm_comm&     pstore_comm,
                                           mpi::rdm_comm&     index_comm,
                                           const int refine_final_n_groups) {
    const int index_k = static_cast<int>(local_index_obj.nids.n_cols());
    // Ranks are paired according to a binary merge tree.
    // Each rank pulls KNNGs from all ranks in the pair partition.
    // Conduct ANN searches from the local points to the pulled KNNGs and merge
    // the results to update the local KNNG.
    // Go to the next level of the tree until the number of partitions becomes
    // refine_final_n_groups.
    int count = 0;
    for (int np = m_comm.size(); np > refine_final_n_groups; np /= 2) {
      const int group_size = m_comm.size() / np;
      const int group_no   = m_comm.rank() / group_size;
      const int pair_group =
          (group_no % 2 == 0) ? (group_no + 1) : (group_no - 1);
      const int    pair_rank_begin = pair_group * group_size;
      const int    pair_rank_end   = pair_rank_begin + group_size;
      const size_t pair_n_points =
          std::accumulate(m_split_sizes.begin() + pair_rank_begin,
                          m_split_sizes.begin() + pair_rank_end, 0ULL);
      spdlog::info("Merge index groups level {}, group size {}, #of groups {}",
                   count, group_size, np);

      rec_time().start("Allocate-recv-buffers-bt-merge");
      spdlog::trace("Allocate receive buffers for merging");
      auto pstores_recv_buf =
          m_nn_driver.alloc_recv_buf(pair_n_points * m_dims * sizeof(fe_type));
      auto knngs_recv_buf =
          m_nn_driver.alloc_recv_buf(pair_n_points * index_k * sizeof(id_type));
      rec_time().stop();

      rec_time().start("Pull-pstore-index-bt-merge");
      std::vector<int> target_ranks(group_size);
      std::iota(target_ranks.begin(), target_ranks.end(), pair_rank_begin);
      std::vector<id_type> id_offsets(group_size + 1, 0);
      for (int i = 0; i < group_size; ++i) {
        id_offsets[i + 1] = id_offsets[i] + m_split_sizes[pair_rank_begin + i];
      }
      for (int i = 0; i < target_ranks.size(); ++i) {
        const int ri = (i + (m_comm.rank() % group_size)) % group_size;
        const int r  = target_ranks[ri];
        assert(r != m_comm.rank());
        priv_initiate_pull_pstore_index(
            r, index_k, pstore_comm, index_comm,
            pstores_recv_buf.get() + id_offsets[ri] * m_dims * sizeof(fe_type),
            knngs_recv_buf.get() + id_offsets[ri] * index_k * sizeof(id_type));
      }
      for (int i = 0; i < target_ranks.size(); ++i) {
        const int ri = (i + (m_comm.rank() % group_size)) % group_size;
        priv_wait_pull_pstore_index(target_ranks[ri], pstore_comm, index_comm);
      }
      rec_time().stop();

      // Optimize pulled KNNG
      spdlog::trace("Optimize pulled KNNG");
      rec_time().start("Optimize-pulled-merged-knng");
      m_nn_driver.optimize_index(index_view_type(
          pair_n_points, index_k, knngs_recv_buf.get(), nullptr));
      rec_time().stop();

      // Run ANN search
      spdlog::trace("Run ANN search");
      const auto trg_pstore =
          pstore_view_type(pair_n_points, m_dims, pstores_recv_buf.get());
      auto trg_index = index_view_type(pair_n_points, index_k,
                                       knngs_recv_buf.get(), nullptr);
      rec_time().start("Run-queries-merge");
      const auto local_pstore = priv_make_local_pstore_view();
      const int  effective_query_k =
          (m_opt.query_k > 0) ? m_opt.query_k : static_cast<int>(m_opt.k);
      auto query_result =
          m_nn_driver.run_queries(local_pstore, trg_pstore, trg_index, false,
                                  m_opt.frontier_size, effective_query_k);
      rec_time().stop();

      rec_time().start("Wait-for-all-queries-bt-merge");
      m_comm.barrier();
      rec_time().stop();

      // -- Update local index with the search results --
      id_type local_index_id_offset;
      id_type query_result_id_offset;
      if (group_no % 2 == 0) {
        local_index_id_offset  = 0;
        query_result_id_offset = m_pid_offsets[pair_rank_begin] -
                                 m_pid_offsets[group_no * group_size];
      } else {
        local_index_id_offset = m_pid_offsets[group_no * group_size] -
                                m_pid_offsets[pair_rank_begin];
        query_result_id_offset = 0;
      }
      rec_time().start("Update-indices-with-qresults-bt-merge");
      m_nn_driver.update_index(query_result, query_result_id_offset,
                               local_index_id_offset, local_index_obj);
      rec_time().stop();

      rec_time().start("Wait-for-all-bt-merge");
      m_comm.barrier();
      rec_time().stop();
    }
  }

  auto priv_merge_and_optimize_group_knng(index_object_type& local_index_obj,
                                          index_nid_type&    group_index_nids,
                                          const int      refine_final_n_groups,
                                          mpi::rdm_comm& single_index_comm) {
    // We first pull all knngs from other ranks in the same group then optimize
    // it for query.
    const int    group_size       = m_comm.size() / refine_final_n_groups;
    const int    group_no         = m_comm.rank() / group_size;
    const int    group_begin_rank = group_no * group_size;
    const int    group_end_rank   = group_begin_rank + group_size;
    const size_t group_n_points =
        std::accumulate(m_split_sizes.begin() + group_begin_rank,
                        m_split_sizes.begin() + group_end_rank, 0ULL);
    const int index_k = static_cast<int>(local_index_obj.nids.n_cols());

    spdlog::trace("Allocate receive buffers for group KNNG");
    rec_time().start("Allocate-recv-buffers-group-knng");
    group_index_nids.reset(group_n_points, index_k);
    auto* knngs_recv_buf = group_index_nids.data();
    rec_time().stop();

    spdlog::trace("Pull KNNGs from ranks in the same group");
    // Pull knngs from all ranks in the same group
    rec_time().start("Pull-group-knng");
    {
      rec_time().start("Launch-get-index");
      for (int offset = 0; offset < group_size; ++offset) {
        // Avoid every rank hitting the same target rank at the same time
        const int trg_rank =
            group_begin_rank +
            (offset + (m_comm.rank() % group_size)) % group_size;
        const size_t buffer_offset =
            (m_pid_offsets[trg_rank] - m_pid_offsets[group_begin_rank]) *
            index_k * sizeof(id_type);
        void* recv_buf =
            reinterpret_cast<unsigned char*>(knngs_recv_buf) + buffer_offset;

        const auto mem_size =
            m_split_sizes[trg_rank] * index_k * sizeof(id_type);
        single_index_comm.async_get(recv_buf, mem_size, trg_rank);
      }
      rec_time().stop();

      rec_time().start("Wait-index");
      for (int offset = 0; offset < group_size; ++offset) {
        const int trg_rank =
            group_begin_rank +
            (offset + (m_comm.rank() % group_size)) % group_size;
        single_index_comm.wait(trg_rank);
      }
      rec_time().stop();
    }
    rec_time().stop();

    // Optimize pulled KNNG
    spdlog::trace("Optimize group KNNG");
    rec_time().start("Optimize-pulled-group-knng");
    m_nn_driver.optimize_index(
        index_view_type(group_n_points, index_k, knngs_recv_buf, nullptr));
    rec_time().stop();

    return index_view_type(group_n_points, index_k, knngs_recv_buf, nullptr);
  }

  void priv_refine_knngs_flat(index_object_type& master_index_obj,
                              mpi::rdm_comm&     pstore_comm,
                              mpi::rdm_comm&     group_index_comm,
                              const int          refine_final_n_groups) {
    const int group_size = m_comm.size() / refine_final_n_groups;
    const int group_no   = m_comm.rank() / group_size;
    // const int group_begin_rank = group_no * group_size;
    const int index_k = static_cast<int>(master_index_obj.nids.n_cols());

    // Allocate receive buffers with the max size.
    size_t max_target_group_n_points = 0;
    {
      for (int target_group_no = 0; target_group_no < refine_final_n_groups;
           ++target_group_no) {
        if (group_no == target_group_no) {
          continue;
        }
        const int    target_rank_begin = target_group_no * group_size;
        const int    target_rank_end   = target_rank_begin + group_size;
        const size_t target_n_points =
            std::accumulate(m_split_sizes.begin() + target_rank_begin,
                            m_split_sizes.begin() + target_rank_end, 0ULL);
        max_target_group_n_points =
            std::max(max_target_group_n_points, target_n_points);
      }
    }
    spdlog::trace("Max target points in refinement: {}",
                  max_target_group_n_points);
    rec_time().start("Allocate-recv-buffers-refine");
    const size_t pstore_recv_buf_size =
        max_target_group_n_points * m_dims * sizeof(fe_type);
    const size_t index_recv_buf_size =
        max_target_group_n_points * index_k * sizeof(id_type);
    spdlog::trace("Allocate a buffer");
    auto pstores_recv_buf = m_nn_driver.alloc_recv_buf(pstore_recv_buf_size);
    auto knngs_recv_buf   = m_nn_driver.alloc_recv_buf(index_recv_buf_size);
#if SALTATLAS_SOLANET_DOUBLE_BUFFERING
    spdlog::trace("Allocate 2nd buffer");
    auto pstores_recv_buf_next = m_nn_driver.alloc_recv_buf(
        (refine_final_n_groups > 2) ? pstore_recv_buf_size : 0);
    auto knngs_recv_buf_next = m_nn_driver.alloc_recv_buf(
        (refine_final_n_groups > 2) ? index_recv_buf_size : 0);
#endif
    rec_time().stop();

    // Prepare target groups info for refinement
    struct refine_target_info {
      int                  target_rank_begin;
      size_t               target_n_points;
      std::vector<int>     target_ranks;
      std::vector<id_type> id_offsets;
    };
    std::vector<refine_target_info> refine_targets_info(refine_final_n_groups);
    for (int target_group_no = 0; target_group_no < refine_final_n_groups;
         ++target_group_no) {
      if (group_no == target_group_no) {
        continue;
      }
      const int          target_rank_begin = target_group_no * group_size;
      const int          target_rank_end   = (target_group_no + 1) * group_size;
      refine_target_info work;
      work.target_rank_begin = target_rank_begin;
      work.target_n_points =
          std::accumulate(m_split_sizes.begin() + target_rank_begin,
                          m_split_sizes.begin() + target_rank_end, 0ULL);
      work.target_ranks.resize(group_size);
      std::iota(work.target_ranks.begin(), work.target_ranks.end(),
                target_rank_begin);
      work.id_offsets.resize(group_size + 1, 0);
      for (int i = 0; i < group_size; ++i) {
        work.id_offsets[i + 1] =
            work.id_offsets[i] + m_split_sizes[target_rank_begin + i];
      }
      refine_targets_info[target_group_no] = std::move(work);
    }

    /// Lambda for launching pull of pstore and index for a target group
    const auto launch_pull_for_target = [&](const refine_target_info& work,
                                            auto* pstore_buf, auto* index_buf) {
      rec_time().start("Pull-pstore-index-refine");
      for (int i = 0; i < work.target_ranks.size(); ++i) {
        // To avoid all ranks in the same group hitting the same target rank
        // at the same time, do a local rank offset within the group.
        const int ri = (i + (m_comm.rank() % group_size)) % group_size;
        const int r  = work.target_ranks[ri];
        assert(r != m_comm.rank());
        priv_initiate_pull_pstore(
            r, pstore_comm,
            pstore_buf + work.id_offsets[ri] * m_dims * sizeof(fe_type));
        if (i == 0) {
          // Index data is already merged as one big KNNG.
          // Pull it onece.
          priv_initiate_pull_index(r, work.target_n_points, index_k,
                                   group_index_comm, index_buf);
        }
      }
      rec_time().stop();
    };

    /// Lambda for waiting for the pull to finish for a target group
    const auto wait_pull_for_target = [&](const refine_target_info& work) {
      rec_time().start("Pull-pstore-index-refine");
      for (int i = 0; i < work.target_ranks.size(); ++i) {
        const int ri = (i + (m_comm.rank() % group_size)) % group_size;
        priv_wait_pull_pstore(work.target_ranks[ri], pstore_comm);
        if (i == 0) {
          // Index data is already merged.
          priv_wait_pull_index(work.target_ranks[ri], group_index_comm);
        }
      }
      rec_time().stop();
    };

    /// Lambda for optimizing the pulled index and running queries for
    /// refinement and updating the local index with the results for a target
    /// group
    const auto refine_target_with_buf = [&](const refine_target_info& work,
                                            auto* pstore_buf, auto* index_buf) {
      const auto trg_pstore =
          pstore_view_type(work.target_n_points, m_dims, pstore_buf);
      auto trg_index =
          index_view_type(work.target_n_points, index_k, index_buf, nullptr);
      const int effective_query_k =
          (m_opt.query_k > 0) ? m_opt.query_k : static_cast<int>(m_opt.k);
      rec_time().start("Run-queries-refine");
      const auto local_pstore = priv_make_local_pstore_view();
      auto       results =
          m_nn_driver.run_queries(local_pstore, trg_pstore, trg_index, false,
                                  m_opt.frontier_size, effective_query_k);
      rec_time().stop();
      rec_time().start("Refine-index-with-queries-results");
      m_nn_driver.update_index(results, m_pid_offsets[work.target_rank_begin],
                               0, master_index_obj);
      rec_time().stop();
    };

    /// The actual refinement process starts here.
    if (!refine_targets_info.empty()) {
      auto get_target_g_no = [&](int offset) {
        return ((offset + group_no) % refine_targets_info.size());
      };
#if SALTATLAS_SOLANET_DOUBLE_BUFFERING
      rec_time().start("Refine-indices-core-double-buffered");
      // Start first pull
      launch_pull_for_target(refine_targets_info[get_target_g_no(1)],
                             pstores_recv_buf.get(), knngs_recv_buf.get());

      for (int i = 1; i < refine_targets_info.size(); ++i) {
        const int target_g_no = get_target_g_no(i);
        assert(target_g_no != group_no);
        const auto& work = refine_targets_info[target_g_no];
        wait_pull_for_target(work);

        if (i + 1 < refine_targets_info.size()) {
          // Pull the next target group's data before refining with the current
          // target's data to hide communication time.
          const int   next_target_g_no = get_target_g_no(i + 1);
          const auto& next_work        = refine_targets_info[next_target_g_no];
          launch_pull_for_target(next_work, pstores_recv_buf_next.get(),
                                 knngs_recv_buf_next.get());
        }
        refine_target_with_buf(work, pstores_recv_buf.get(),
                               knngs_recv_buf.get());
        std::swap(pstores_recv_buf, pstores_recv_buf_next);
        std::swap(knngs_recv_buf, knngs_recv_buf_next);
      }
      rec_time().stop();
#else
      rec_time().start("Refine-indices-core");
      for (int i = 1; i < refine_targets_info.size(); ++i) {
        const int target_g_no = get_target_g_no(i);
        assert(target_g_no != group_no);
        const auto& work = refine_targets_info[target_g_no];

        launch_pull_for_target(work, pstores_recv_buf.get(),
                               knngs_recv_buf.get());
        wait_pull_for_target(work);
        refine_target_with_buf(work, pstores_recv_buf.get(),
                               knngs_recv_buf.get());
      }
      rec_time().stop();
#endif
    }
    rec_time().start("Wait-for-all-refine-to-finish");
    m_comm.barrier();
    rec_time().stop();
  }

  auto priv_initiate_pull_pstore(const int trg_rank, mpi::rdm_comm& pstore_win,
                                 void* pstore_buf) {
    const auto trg_n_points = m_split_sizes[trg_rank];
    const auto mem_size     = trg_n_points * m_dims * sizeof(fe_type);
    rec_time().start("Launch-get-pstore");
    pstore_win.async_get(pstore_buf, mem_size, trg_rank);
    rec_time().stop();
  }

  auto priv_initiate_pull_index(const int trg_rank, const size_t index_k,
                                mpi::rdm_comm& index_win, void* index_buf) {
    const auto trg_n_points = m_split_sizes[trg_rank];
    const auto mem_size     = trg_n_points * index_k * sizeof(id_type);
    rec_time().start("Launch-get-index");
    index_win.async_get(index_buf, mem_size, trg_rank);
    rec_time().stop();
  }

  auto priv_initiate_pull_index(const int trg_rank, const size_t trg_n_points,
                                const size_t index_k, mpi::rdm_comm& index_win,
                                void* index_buf) {
    const auto mem_size = trg_n_points * index_k * sizeof(id_type);
    rec_time().start("Launch-get-index");
    index_win.async_get(index_buf, mem_size, trg_rank);
    rec_time().stop();
  }

  auto priv_initiate_pull_pstore_index(const int trg_rank, const size_t index_k,
                                       mpi::rdm_comm& pstore_win,
                                       mpi::rdm_comm& index_win,
                                       void* pstore_buf, void* index_buf) {
    priv_initiate_pull_pstore(trg_rank, pstore_win, pstore_buf);
    priv_initiate_pull_index(trg_rank, index_k, index_win, index_buf);
  }

  void priv_wait_pull_pstore(const int trg_rank, mpi::rdm_comm& pstore_win) {
    rec_time().start("Wait-pstore");
    pstore_win.wait(trg_rank);
    rec_time().stop();
  }

  void priv_wait_pull_index(const int trg_rank, mpi::rdm_comm& index_win) {
    rec_time().start("Wait-index");
    index_win.wait(trg_rank);
    rec_time().stop();
  }

  void priv_wait_pull_pstore_index(const int      trg_rank,
                                   mpi::rdm_comm& pstore_win,
                                   mpi::rdm_comm& index_win) {
    priv_wait_pull_pstore(trg_rank, pstore_win);
    priv_wait_pull_index(trg_rank, index_win);
  }

  // Make global ID map that maps offset-based IDs to global IDs
  std::vector<id_type> priv_make_global_o2e_id_maps() const {
    // Make global offset-to-global ID map first.
    spdlog::trace("Const ID map");
    rec_time().start("Const-o2e-id-map");
    const size_t n_total_local_points =
        std::accumulate(m_split_sizes.begin(), m_split_sizes.end(), size_t{0});
    std::vector<id_type> global_o2e_id_map(n_total_local_points);
    rec_time().stop();

    // Gather all l2e ID maps from all ranks and write each received chunk
    // directly to the final map to avoid holding an extra global copy.
    spdlog::trace("Gather all l2e ID maps");
    rec_time().start("Gather-l2e-id-maps");
    auto id_map_exchanger = [&, this](const int pair) {
      std::vector<id_type> id_map_recv;
      m_comm.sendrecv_arb_size(pair, m_l2e_id_table, id_map_recv);

      assert(pair >= 0);
      assert(static_cast<size_t>(pair) < m_split_sizes.size());
      assert(id_map_recv.size() == m_split_sizes[pair]);

      const auto offset = m_pid_offsets[pair];
      OMP_DIRECTIVE(parallel for)
      for (size_t lpid = 0; lpid < id_map_recv.size(); ++lpid) {
        const auto o_pid = lpid + offset;
        assert(o_pid < global_o2e_id_map.size());
        global_o2e_id_map[o_pid] = id_map_recv[lpid];
      }
    };
    mpi::pair_wise_all_to_all(m_comm.size(), m_comm.rank(), id_map_exchanger,
                              m_comm.comm(), false);
    rec_time().stop();

    return global_o2e_id_map;
  }

  sparse_knng_type priv_conv_to_final_graph(
      const std::vector<id_type>& global_o2e_id_map,
      index_object_type&          master_index_obj) {
    sparse_knng_type final_knng;
    auto             n_points = m_l2e_id_table.size();
    final_knng.reserve(n_points);

    rec_time().start("Copy-master-knng2host");
    // Master index
    // Holds the best known neighbors so far.
    // Neighbors are not limited to local points. Their IDs are the
    // offset-based ones.
    index_view_type host_master_index =
        m_nn_driver.make_index_view(master_index_obj);
    rec_time().stop();

    rec_time().start("Allocate-final-knng");
    // Allocate space for the final knng first
    for (size_t lpid = 0; lpid < n_points; ++lpid) {
      const auto e_pid = m_l2e_id_table.at(lpid);
      final_knng[e_pid].reserve(m_opt.k);
    }
    rec_time().stop();

    rec_time().start("Add-edges-to-final-knng");
    OMP_DIRECTIVE(parallel for)
    for (size_t lpid = 0; lpid < n_points; ++lpid) {
      const auto e_pid = m_l2e_id_table.at(lpid);
      auto       nids  = host_master_index.neighbor_ids(lpid);
      auto       dists = host_master_index.neighbor_dists(lpid);
      for (size_t j = 0; j < m_opt.k; ++j) {
        const auto o_nid = nids[j];
        if (o_nid >= global_o2e_id_map.size()) {
          m_comm.cerr() << j << " Offset-based neighbor ID " << o_nid
                        << " is out of range. Max allowed ID is "
                        << (global_o2e_id_map.size() - 1)
                        << ". Distance: " << dists[j] << std::endl;
          m_comm.abort();
        }

        const auto      e_nid = global_o2e_id_map[o_nid];
        const dist_type dist  = dists[j];
        final_knng[e_pid].emplace_back(e_nid, dist);
      }
    }
    rec_time().stop();

    return final_knng;
  }

  mpi::rdm_comm priv_create_mpi_win_for_pstore() {
    rec_time().start("Setup-rdma-pstore");
    auto*         buf  = m_pstore_obj.data();
    const auto    size = m_l2e_id_table.size() * m_dims * sizeof(fe_type);
    mpi::rdm_comm rdm_comm(m_comm, buf, size, MPI_INFO_NULL);
    rec_time().stop();
    return rdm_comm;
  }

  pstore_view_type priv_make_local_pstore_view() {
    return pstore_view_type(m_l2e_id_table.size(), m_dims, m_pstore_obj.data());
  }

  mpi::rdm_comm priv_create_mpi_win_for_index(void* buf, const size_t size) {
    rec_time().start("Setup-rdma-index");
    mpi::rdm_comm rdm_comm(m_comm, buf, size, MPI_INFO_NULL);
    rec_time().stop();
    return rdm_comm;
  }

  // For debuging
  void priv_show_dram_usage(const size_t line_no,
                            const bool   barrier = true) const {
    if (barrier) {
      m_comm.barrier();
    }
    if (m_opt.verbose) {
      m_comm.cout0() << "DRAM usages at line " << line_no << std::endl;
      m_comm.cout0() << "DRAM used (GiB): "
                     << metall::mtlldetail::get_used_ram_size() /
                            double(1 << 30)
                     << std::endl;
      m_comm.cout0() << "DRAM free (GiB): "
                     << metall::mtlldetail::get_free_ram_size() /
                            double(1 << 30)
                     << std::endl;
      m_comm.cout0() << "DRAM cache (GiB): "
                     << metall::mtlldetail::get_page_cache_size() /
                            double(1 << 30)
                     << std::endl;
    }
    if (barrier) {
      m_comm.barrier();
    }
  }

  options            m_opt;
  mpi::communicator& m_comm;
  size_t             m_dims{0};
  size_t             m_g_n_points{0};  // #of totall points globally

  // Local NN drivers
  nn_driver m_nn_driver;

  pstore_object_type m_pstore_obj;

  // Sizes of splits (one split per rank)
  std::vector<size_t> m_split_sizes;
  // ID offsets (one offset value per rank)
  // Used to convert local point IDs to global offset-based IDs
  std::vector<size_t> m_pid_offsets;

  // Local ID (not offset-based one) -> external ID tables
  // use vector for each table to send over MPI efficiently
  std::vector<id_type> m_l2e_id_table;
};  // namespace

/// \brief Optimize kNNG. Specifically, make the graph undirected and prune
/// high-degree vertices.. This is done in-place.
/// \param knng The kNNG to optimize. It is modified in-place to save memory. It
/// is expected to contain distances as well.
template <typename id_type, typename dist_type, typename knng_type,
          typename partitioner_type>
void optimize_knng(knng_type& knng, const partitioner_type& partitioner,
                   mpi::communicator& comm, const size_t max_degree,
                   const bool verbose) {
  spdlog::info("Optimize graph");
  rec_time().start("Optimize");

  spdlog::trace("Make rKNNG locally");
  std::vector<id_type> local_sids;
  local_sids.reserve(knng.size());
  for (const auto& item : knng) {
    local_sids.push_back(item.first);
  }
  assert(local_sids.size() == knng.size());

  // Holds reversed graph data
  // Each thread has its own independent table to avoid race condition
  std::vector<std::vector<std::vector<id_type>>>   g_r_nids;
  std::vector<std::vector<std::vector<dist_type>>> g_r_dits;

  int n_threads = 0;
  OMP_DIRECTIVE(parallel) {
    n_threads = omp::get_num_threads();
    OMP_DIRECTIVE(single) {
      g_r_nids.resize(n_threads);
      g_r_dits.resize(n_threads);
    }
    OMP_DIRECTIVE(barrier)

    const int tid    = omp::get_thread_num();
    auto&     r_nids = g_r_nids.at(tid);
    auto&     r_dits = g_r_dits.at(tid);
    r_nids.resize(comm.size());
    r_dits.resize(comm.size());

    OMP_DIRECTIVE(for)
    for (size_t si = 0; si < local_sids.size(); ++si) {
      const auto sid = local_sids[si];
      for (const auto& neighbor : knng.at(sid)) {
        const auto nid      = neighbor.id;
        const auto dit      = neighbor.distance;
        const auto dst_rank = partitioner(nid);
        r_nids.at(dst_rank).push_back(nid);
        r_nids.at(dst_rank).push_back(sid);
        r_dits.at(dst_rank).push_back(dit);
      }
    }
  }
  comm.barrier();

  spdlog::trace("Exchange rKNNG");
  std::vector<std::mutex> mutexes(n_threads * 8);
  auto                    graph_exchanger = [&](const int pair_rank) {
    for (int tid = 0; tid < n_threads; ++tid) {
      auto&                r_nids = g_r_nids.at(tid);
      std::vector<id_type> recv_nids;
      comm.sendrecv_arb_size_opt(pair_rank, std::move(r_nids[pair_rank]),
                                                    recv_nids);

      auto&                  r_dits = g_r_dits.at(tid);
      std::vector<dist_type> recv_dists;
      comm.sendrecv_arb_size_opt(pair_rank, std::move(r_dits[pair_rank]),
                                                    recv_dists);
      assert(recv_nids.size() == recv_dists.size() * 2);

      // Merge rKNNG to KNNG
      OMP_DIRECTIVE(parallel for)
      for (std::size_t i = 0; i < recv_nids.size() / 2; ++i) {
        const auto                  r_sid = recv_nids[i * 2];
        const auto                  r_nid = recv_nids[i * 2 + 1];
        std::lock_guard<std::mutex> lock(mutexes[r_sid % mutexes.size()]);
        knng.at(r_sid).emplace_back(r_nid, recv_dists[i]);
      }
    }
  };
  mpi::pair_wise_all_to_all(comm.size(), comm.rank(), graph_exchanger,
                            comm.comm(), false);

  std::size_t actual_max_degree = 0;
  // Sort neighbors by distances and prune high-degree points
  spdlog::trace("Optimize KNNG locally");

  OMP_DIRECTIVE(parallel for reduction(max: actual_max_degree))
  for (size_t si = 0; si < local_sids.size(); ++si) {
    auto& neighbors = knng.at(local_sids[si]);

    // remove duplicates
    std::sort(neighbors.begin(), neighbors.end(),
              [](const auto& x, const auto& y) { return x.id < y.id; });
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end(),
                                [](const auto& lhd, const auto& rhd) {
                                  return lhd.id == rhd.id;
                                }),
                    neighbors.end());

    // Sort neighbors by distance
    std::sort(
        neighbors.begin(), neighbors.end(),
        [](const auto& x, const auto& y) { return x.distance < y.distance; });

    // Prune high-degree points
    if (neighbors.size() > max_degree) {
      neighbors.resize(max_degree);
    }
    actual_max_degree = std::max(neighbors.size(), actual_max_degree);
  }
  comm.barrier();
  spdlog::info("Finished optimization");
  if (verbose) {
    comm.cout0() << "Max degree: " << comm.all_reduce_max(actual_max_degree)
                 << std::endl;
  }
  rec_time().stop();
}

std::string gen_knng_file_name(const std::filesystem::path& dir_path,
                               const int                    rank) {
  return dir_path / (std::string("knng-") + std::to_string(rank));
}

/// \brief Dump kNNG to files. Each rank dumps its own part of the kNNG to a
/// file. The file name is determined by the rank and the given directory path.
template <typename sparse_knng_type>
inline void dump_knng(const std::filesystem::path& dir_path,
                      const sparse_knng_type&      knng,
                      const mpi::communicator&     comm,
                      const bool                   dump_distance = false) {
  if (comm.rank() == 0) {
    std::filesystem::create_directories(dir_path);
  }
  comm.barrier();

  std::filesystem::path knng_out_path(
      gen_knng_file_name(dir_path, comm.rank()));
  saltatlas::utility::dump_knng(knng_out_path, knng, dump_distance);
  comm.barrier();
}

/// \brief Load kNNG from files. Each rank loads its own part of the kNNG from a
/// file. The file name is determined by the rank and the given directory path.
/// \note This function does not repartition the loaded kNNG.
template <typename sparse_knng_type>
inline void load_knng(const std::filesystem::path& knng_dir_path,
                      const mpi::communicator& comm, sparse_knng_type& knng,
                      const bool has_distance = false) {
  std::filesystem::path knng_path(
      gen_knng_file_name(knng_dir_path, comm.rank()));
  saltatlas::utility::load_knng(knng_path, knng, has_distance);
  comm.barrier();
}
}  // namespace saltatlas::solanet
