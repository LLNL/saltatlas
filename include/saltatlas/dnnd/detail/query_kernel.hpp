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
#include <saltatlas/dnnd/detail/utilities/allocator.hpp>
#include <saltatlas/dnnd/detail/utilities/general.hpp>
#include <saltatlas/dnnd/detail/utilities/mpi.hpp>
#include "saltatlas/point_store.hpp"

namespace saltatlas::dndetail {

template <typename PointStore, typename KNNIndex>
class dknn_batch_query_kernel {
  static_assert(
      std::is_same_v<typename PointStore::id_type, typename KNNIndex::id_type>,
      "ID types do not match.");

 public:
  using id_type       = typename PointStore::id_type;
  using distance_type = typename KNNIndex::distance_type;
  using point_type    = typename PointStore::point_type;

  // Redefine point store type so that autocompletion works when writing code.
  using point_store_type =
      point_store<id_type, point_type, typename PointStore::hasher,
                  typename PointStore::equal_to,
                  typename PointStore::allocator_type>;
  // Redefine index store type so that autocompletion works when writing code.
  using nn_index_type =
      nn_index<id_type, distance_type, typename KNNIndex::allocator_type>;

  using point_partitioner = std::function<int(const id_type& id)>;
  using distance_function_type =
      distance::distance_function_type<point_type, distance_type>;
  using neighbor_type = typename nn_index_type::neighbor_type;

  // These data stores are allocated on DRAM
  using query_store_type    = std::vector<point_type>;
  using neighbor_store_type = std::vector<std::vector<neighbor_type>>;

  struct option {
    int         k{4};
    double      epsilon{0.1};
    double      mu{0.0};
    std::size_t batch_size{0};
    uint64_t    rnd_seed{128};
    bool        verbose{false};
  };

  dknn_batch_query_kernel(const option&                 opt,
                          const point_store_type&       point_store,
                          const point_partitioner&      partitioner,
                          const distance_function_type& distance_function,
                          const nn_index_type& nn_index, ygm::comm& comm)
      : m_option(opt),
        m_point_store(point_store),
        m_point_partitioner(partitioner),
        m_distance_function(distance_function),
        m_nn_index(nn_index),
        m_comm(comm),
        m_rnd_generator(m_option.rnd_seed + m_comm.rank()) {
    priv_find_max_id();
    m_self.check(m_comm);
  }

  void query_batch(const query_store_type& queries,
                   neighbor_store_type&    query_results) {
    priv_query_batch(queries, query_results);
  }

  ygm::comm& comm() { return m_comm; }

 private:
  using self_type           = dknn_batch_query_kernel<PointStore, KNNIndex>;
  using self_pointer_type   = ygm::ygm_ptr<self_type>;
  using knn_heap_type       = unique_knn_heap<id_type, distance_type>;
  using knn_heap_table_type = std::unordered_map<std::size_t, knn_heap_type>;
  using internal_query_store_type = point_store<id_type, point_type>;

  void priv_find_max_id() {
    m_global_max_id = 0;
    for (const auto& [id, _] : m_point_store) {
      m_global_max_id = std::max(m_global_max_id, id);
    }
    m_global_max_id = m_comm.all_reduce_max(m_global_max_id);
  }

  id_type priv_all_gather_query(const query_store_type& queries) {
    const std::size_t max_num_queries = m_comm.all_reduce_max(queries.size());
    const id_type     offset          = max_num_queries * m_comm.rank();

    // Allgather
    for (int r = 0; r < m_comm.size(); ++r) {
      for (std::size_t i = 0; i < queries.size(); ++i) {
        m_comm.async(
            r,
            [](const ygm::ygm_ptr<self_type>& dst_self, const id_type id,
               const point_type& q) {
              assert(!dst_self->m_query_store.contains(id));
              dst_self->m_query_store[id] = q;
            },
            m_self, i + offset, queries[i]);
      }
    }
    m_comm.barrier();

    // Local ID offset.
    // Query IDs are sequentially increased within locals.
    return offset;
  }

  /// Assumes that queries are already partitioned.
  void priv_query_batch(const query_store_type& queries,
                        neighbor_store_type&    query_results) {
    const auto local_query_no_offset = priv_all_gather_query(queries);

    const std::size_t num_global_queries = m_query_store.size();
    if (m_option.verbose) {
      m_comm.cout0() << "#of queries\t" << num_global_queries << std::endl;
      m_comm.cout0() << "Batch Size\t" << m_option.batch_size << std::endl;
    }

    std::size_t query_no_offset   = local_query_no_offset;
    std::size_t num_local_remains = queries.size();
    for (std::size_t batch_no = 0;; ++batch_no) {
      if (m_option.verbose) {
        m_comm.cout0() << "\n[Batch No. " << batch_no << "]" << std::endl;
      }

      const auto local_batch_size =
          mpi::assign_tasks(num_local_remains, m_option.batch_size,
                            m_comm.rank(), m_comm.size(), m_option.verbose);

      // Initialize the variable used during the query.
      m_knn_heap_table.clear();
      m_visited.clear();
      for (std::size_t i = 0; i < local_batch_size; ++i) {
        const auto        query_no  = query_no_offset + i;
        const std::size_t heap_size = m_option.k * (1.0 + m_option.mu);
        m_knn_heap_table.emplace(query_no, heap_size);
        m_visited[query_no].clear();
      }
      m_comm.cf_barrier();

      // Launch queries.
      for (std::size_t i = 0; i < local_batch_size; ++i) {
        const auto query_no = query_no_offset + i;
        priv_launch_asynch_single_query(query_no, m_rnd_generator);
        --num_local_remains;
      }
      m_comm.barrier();

      // Move query results from heap to an array-like data structure.
      for (std::size_t i = 0; i < local_batch_size; ++i) {
        const auto query_no = query_no_offset + i;
        auto&      heap     = m_knn_heap_table.at(query_no);
        if (heap.empty()) {
          m_comm.cerr0() << query_no << "-th query result is empty."
                         << std::endl;
          MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        query_results.resize(query_results.size() + 1);
        auto& result_knn = query_results.back();
        while (!heap.empty()) {
          result_knn.push_back(heap.top());
          heap.pop();
        }
        std::partial_sort(result_knn.begin(), result_knn.begin() + m_option.k,
                          result_knn.end());
        result_knn.resize(m_option.k);  // Need only nearest k neighbors
      }
      query_no_offset += local_batch_size;
      m_comm.cf_barrier();

      const auto num_global_remains = m_comm.all_reduce_sum(num_local_remains);
      if (num_global_remains == 0) break;
      if (m_option.verbose) {
        m_comm.cout0() << "#of remaining queries\t" << num_global_remains
                       << std::endl;
      }
    }  // end of all queries
    if (m_comm.all_reduce_sum(num_local_remains) > 0) {
      m_comm.cout0() << "Logic error!! Not all queries have been processed"
                     << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    if (m_option.verbose) {
      m_comm.cout0() << "Finished all queries" << std::endl;
    }
  }

  /// Launches a single query asynchronously.
  /// The query result is stored to m_knn_heap_table in the owner of the query.
  template <typename random_generator_type>
  void priv_launch_asynch_single_query(const std::size_t      query_no,
                                       random_generator_type& rnd_gen) {
    std::unordered_set<id_type>            set;
    std::uniform_int_distribution<id_type> dis(0, m_global_max_id);

    // Visit randomly selected points as search starting points.
    for (std::size_t k = 0; k < m_knn_heap_table.at(query_no).k();
         ++k) {  // sqrt(k) is enough?
      while (true) {
        const id_type id = dis(rnd_gen);
        if (set.count(id)) continue;
        set.insert(id);

        assert(m_point_partitioner);
        m_comm.async(m_point_partitioner(id), neighbor_visitor_launcher{},
                     m_self, m_comm.rank(), query_no, id,
                     std::numeric_limits<distance_type>::max());
        break;
      }
    }
  }

  struct neighbor_visitor_launcher {
    void operator()(const self_pointer_type& local_this,
                    const int query_owner_rank, const std::size_t query_no,
                    const id_type src_id, const distance_type max_distance) {
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
    void operator()(const self_pointer_type& local_this,
                    const int query_owner_rank, const std::size_t query_no,
                    const id_type trg_id, const distance_type& max_distance) {
      const auto& query_point = local_this->m_query_store[query_no];
      assert(local_this->m_point_store.contains(trg_id));
      const auto& trg_point = local_this->m_point_store[trg_id];
      const auto  d = local_this->m_distance_function(query_point, trg_point);
      if (d >= max_distance) return;

      local_this->comm().async(query_owner_rank, neighbor_updator{}, local_this,
                               query_no, trg_id, d);
    }
  };

  struct neighbor_updator {
    void operator()(const self_pointer_type& local_this,
                    const std::size_t query_no, const id_type nid,
                    const distance_type d) {
      assert(local_this->m_visited.count(query_no));
      if (local_this->m_visited[query_no].count(nid)) {
        return;  // Already visited
      }
      local_this->m_visited[query_no].insert(nid);

      knn_heap_type& heap = local_this->m_knn_heap_table.at(query_no);
      // m_visited does the same work
      //      if (heap.contains(nid)) {
      //        return;  // Already one the nearest neighbors
      //      }

      auto max_distance =
          heap.size() < heap.k()
              ? std::numeric_limits<distance_type>::max()
              : heap.top().distance * (1.0 + local_this->m_option.epsilon);
      if (d >= max_distance) {
        return;  // Too far neighbor.
      }

      if (heap.push_unique(nid, d)) {
        max_distance =
            heap.size() < heap.k()
                ? std::numeric_limits<distance_type>::max()
                : heap.top().distance * (1.0 + local_this->m_option.epsilon);
      }

      const auto& partitioner = local_this->m_point_partitioner;
      assert(partitioner);
      local_this->comm().async(partitioner(nid), neighbor_visitor_launcher{},
                               local_this, local_this->comm().rank(), query_no,
                               nid, max_distance);
    }
  };

  option                        m_option;
  const point_store_type&       m_point_store;
  const point_partitioner       m_point_partitioner;
  const distance_function_type& m_distance_function;
  const nn_index_type&          m_nn_index;
  ygm::comm&                    m_comm;
  self_pointer_type             m_self{this};
  knn_heap_table_type           m_knn_heap_table;
  id_type                       m_global_max_id{0};
  internal_query_store_type     m_query_store;
  std::mt19937                  m_rnd_generator;
  std::unordered_map<std::size_t, std::unordered_set<id_type>> m_visited;
};

}  // namespace saltatlas::dndetail