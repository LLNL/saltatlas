// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <optional>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/detail/distance.hpp>
#include <saltatlas/dnnd/detail/neighbor.hpp>
#include <saltatlas/dnnd/detail/nn_index.hpp>
#include <saltatlas/dnnd/detail/point_store.hpp>
#include <saltatlas/dnnd/detail/utilities/allocator.hpp>
#include <saltatlas/dnnd/detail/utilities/general.hpp>
#include <saltatlas/dnnd/detail/utilities/mpi.hpp>

namespace saltatlas::dndetail {

template <typename PointStore, typename KNNIndex>
class dknn_batch_query_kernel {
  static_assert(
      std::is_same_v<typename PointStore::id_type, typename KNNIndex::id_type>,
      "ID types do not match.");

 public:
  using id_type              = typename PointStore::id_type;
  using distance_type        = typename KNNIndex::distance_type;
  using feature_element_type = typename PointStore::feature_element_type;

  // Redefine point store type so that autocompletion works when writing code.
  using point_store_type = point_store<id_type, feature_element_type,
                                       typename PointStore::allocator_type>;
  // Redefine index store type so that autocompletion works when writing code.
  using nn_index_type =
      nn_index<id_type, distance_type, typename KNNIndex::allocator_type>;

  using point_partitioner = std::function<int(const id_type& id)>;
  using distance_metric =
      distance::metric_type<feature_element_type, distance_type>;
  using neighbor_type = typename nn_index_type::neighbor_type;

  // These data stores are allocated on DRAM
  using query_point_store_type = point_store<id_type, feature_element_type>;
  using knn_store_type =
      std::unordered_map<id_type, std::vector<neighbor_type>>;

  struct option {
    int         k{4};
    std::size_t batch_size{0};
    uint64_t    rnd_seed{128};
    bool        verbose{false};
  };

  dknn_batch_query_kernel(const option&            opt,
                          const point_store_type&  point_store,
                          const point_partitioner& partitioner,
                          const distance_metric&   metric,
                          const nn_index_type& nn_index, ygm::comm& comm)
      : m_option(opt),
        m_point_store(point_store),
        m_point_partitioner(partitioner),
        m_distance_metric(metric),
        m_nn_index(nn_index),
        m_comm(comm),
        m_rnd_generator(m_option.rnd_seed + m_comm.rank()) {
    m_global_max_id = m_comm.all_reduce_max(m_point_store.max_id());
    m_comm.cf_barrier();
    m_this.check(m_comm);
  }

  /// Assumes that all processes have the same query_points data.
  void query_batch(const query_point_store_type& query_points,
                   knn_store_type&               query_result) {
    priv_query_batch(query_points, query_result);
  }

  ygm::comm& comm() { return m_comm; }

 private:
  using self_type           = dknn_batch_query_kernel<PointStore, KNNIndex>;
  using self_pointer_type   = ygm::ygm_ptr<self_type>;
  using knn_heap_type       = unique_knn_heap<id_type, distance_type>;
  using knn_heap_table_type = std::unordered_map<std::size_t, knn_heap_type>;

  void priv_query_batch(const query_point_store_type& query_points,
                        knn_store_type&               query_result) {
    m_query_points = query_points;

    const std::size_t num_global_queries = query_points.size();
    const auto        range =
        partial_range(num_global_queries, m_comm.rank(), m_comm.size());

    if (m_option.verbose) {
      m_comm.cout0() << "#of queries\t" << num_global_queries << std::endl;
      m_comm.cout0() << "Batch Size\t" << m_option.batch_size << std::endl;
    }

    std::size_t query_no_offset     = range.first;
    std::size_t last_query_no       = range.second - 1;
    std::size_t count_local_queries = 0;
    for (std::size_t batch_no = 0;; ++batch_no) {
      if (m_option.verbose) {
        m_comm.cout0() << "\n[Batch No. " << batch_no << "]" << std::endl;
      }

      const auto local_batch_size = mpi::assign_tasks(
          last_query_no - query_no_offset + 1, m_option.batch_size,
          m_comm.rank(), m_comm.size(), m_option.verbose);

      m_knn_heap_table.clear();
      for (std::size_t i = 0; i < local_batch_size; ++i) {
        const auto query_no = query_no_offset + i;
        m_knn_heap_table.emplace(query_no, m_option.k);
      }
      m_comm.cf_barrier();

      for (std::size_t i = 0; i < local_batch_size; ++i) {
        const auto query_no = query_no_offset + i;
        priv_launch_asynch_single_query(query_no, m_rnd_generator);
        ++count_local_queries;
      }
      m_comm.barrier();

      // Convert the query results
      for (std::size_t i = 0; i < local_batch_size; ++i) {
        const auto query_no = query_no_offset + i;
        auto&      knn      = query_result[query_no];

        auto& heap = m_knn_heap_table.at(query_no);
        if (heap.empty()) {
          std::cerr << query_no << "-th knn heap is empty." << std::endl;
          MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        while (!heap.empty()) {
          knn.push_back(heap.top());
          heap.pop();
        }
        std::sort(knn.begin(), knn.end());
      }
      query_no_offset += local_batch_size;
      m_comm.cf_barrier();

      const auto global_remaining_queries =
          m_comm.all_reduce_sum(last_query_no - query_no_offset + 1);
      if (global_remaining_queries == 0) break;
      if (m_option.verbose) {
        m_comm.cout0() << "#of remaining queries\t" << global_remaining_queries
                       << std::endl;
      }
    }
    if (m_comm.all_reduce_sum(count_local_queries) != num_global_queries) {
      m_comm.cout0() << "Logic error!! Not all queries have been processed"
                     << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  template <typename random_generator_type>
  void priv_launch_asynch_single_query(const std::size_t      query_no,
                                       random_generator_type& rnd_gen) {
    std::unordered_set<id_type>            set;
    std::uniform_int_distribution<id_type> dis(0, m_global_max_id);
    for (std::size_t k = 0; k < m_option.k; ++k) {  // sqrt(k) is enough?
      while (true) {
        const id_type id = dis(rnd_gen);
        if (set.count(id)) continue;
        set.insert(id);
        assert(m_point_partitioner);
        m_comm.async(m_point_partitioner(id), neighbor_visitor_launcher{},
                     m_this, m_comm.rank(), query_no, id,
                     std::numeric_limits<distance_type>::max());
        break;
      }
    }
  }

  struct neighbor_visitor_launcher {
    void operator()(self_pointer_type local_this, const int query_owner_rank,
                    const std::size_t query_no, const id_type src_id,
                    const distance_type max_distance) {
      const auto& nn_index    = local_this->m_nn_index;
      const auto& partitioner = local_this->m_point_partitioner;
      assert(partitioner);
      if (nn_index.num_neighbors(src_id) == 0) {
        std::cerr << "Point " << src_id << " has no neighbors" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      for (auto nitr = nn_index.neighbors_begin(src_id),
                end  = nn_index.neighbors_end(src_id);
           nitr != end; ++nitr) {
        const auto& nid = nitr->id;
        local_this->comm().async(partitioner(nid), distance_calculator{},
                                 local_this, query_owner_rank, query_no, nid,
                                 max_distance);
      }
    }
  };

  struct distance_calculator {
    void operator()(self_pointer_type local_this, const int query_owner_rank,
                    const std::size_t query_no, const id_type trg_id,
                    const distance_type& max_distance) {
      const auto& query_feature =
          local_this->m_query_points->feature_vector(query_no);
      assert(local_this->m_point_store.contains(trg_id));
      const auto& trg_feature =
          local_this->m_point_store.feature_vector(trg_id);
      const auto d = local_this->m_distance_metric(
          query_feature.size(), query_feature.data(), trg_feature.data());
      if (d >= max_distance) return;

      local_this->comm().async(query_owner_rank, neighbor_updator{}, local_this,
                               query_no, trg_id, d);
    }
  };

  struct neighbor_updator {
    void operator()(self_pointer_type local_this, const std::size_t query_no,
                    const id_type nid, const distance_type d) {
      knn_heap_type& heap = local_this->m_knn_heap_table.at(query_no);
      if (!heap.push_unique(nid, d)) {
        return;
      }
      const auto& partitioner = local_this->m_point_partitioner;
      assert(partitioner);
      const auto max_distance = heap.size() < local_this->m_option.k
                                    ? std::numeric_limits<distance_type>::max()
                                    : heap.top().distance;
      local_this->comm().async(partitioner(nid), neighbor_visitor_launcher{},
                               local_this, local_this->comm().rank(), query_no,
                               nid, max_distance);
    }
  };

  option                                m_option;
  const point_store_type&               m_point_store;
  const point_partitioner               m_point_partitioner;
  const distance_metric&                m_distance_metric;
  const nn_index_type&                  m_nn_index;
  ygm::comm&                            m_comm;
  self_pointer_type                     m_this{this};
  knn_heap_table_type                   m_knn_heap_table;
  id_type                               m_global_max_id{0};
  std::optional<query_point_store_type> m_query_points{std::nullopt};
  std::mt19937                          m_rnd_generator;
};

}  // namespace saltatlas::dndetail