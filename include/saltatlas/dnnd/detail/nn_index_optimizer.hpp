// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <functional>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

#include <saltatlas/dnnd/detail/distance.hpp>
#include <saltatlas/dnnd/detail/neighbor.hpp>
#include <saltatlas/dnnd/detail/neighbor_cereal.hpp>
#include <saltatlas/dnnd/detail/nn_index.hpp>
#include <saltatlas/dnnd/detail/point_store.hpp>

namespace saltatlas::dndetail {

template <typename PointStore, typename KNNIndex>
class nn_index_optimizer {
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

  using feature_vector_type = typename point_store_type::feature_vector_type;
  using point_partitioner   = std::function<int(const id_type& id)>;
  using distance_metric =
      distance::metric_type<feature_element_type, distance_type>;
  using neighbor_type = typename nn_index_type::neighbor_type;

  struct option {
    std::size_t index_k{0};
    bool        undirected{false};
    double      pruning_degree_multiplier{-1};  // if <= 0, no pruning.
    bool        remove_long_paths{false};
    std::size_t batch_size{0};  // if <= 0, process all at a time.
    bool        verbose{false};
  };

  nn_index_optimizer(const option& opt, const point_store_type& point_store,
                     const point_partitioner& partitioner,
                     const distance_metric& metric, nn_index_type& nn_index,
                     ygm::comm& comm)
      : m_option(opt),
        m_point_store(point_store),
        m_point_partitioner(partitioner),
        m_distance_metric(metric),
        m_nn_index(nn_index),
        m_comm(comm) {
    m_this.check(m_comm);
  }

  void run() {
    if (m_option.undirected) {
      priv_make_index_undirected();
    }
    if (m_option.pruning_degree_multiplier > 0) {
      priv_prune_neighbors();
    }
    if (m_option.remove_long_paths) {
      priv_remove_long_paths();
    }
  }

 private:
  using self_type         = nn_index_optimizer<PointStore, KNNIndex>;
  using self_pointer_type = ygm::ygm_ptr<self_type>;

  static constexpr std::size_t k_batch_size = 1000000;

  void priv_make_index_undirected() {
    if (m_option.verbose) {
      m_comm.cout0() << "Making the index undirected" << std::endl;
    }
    auto        reversed_index        = priv_generate_reverse_index();
    std::size_t total_local_neighbors = 0;
    std::size_t max_degree            = 0;
    for (auto pitr = reversed_index.points_begin(),
              pend = reversed_index.points_end();
         pitr != pend; ++pitr) {
      const auto& source = pitr->first;
      assert(m_point_partitioner(source) == m_comm.rank());
      for (auto nitr = reversed_index.neighbors_begin(source),
                nend = reversed_index.neighbors_end(source);
           nitr != nend; ++nitr) {
        m_nn_index.insert(source, *nitr);
      }
      reversed_index.reset_neighbors(source);  // release memory.
      m_nn_index.sort_and_remove_duplicate_neighbors(source);
      total_local_neighbors += m_nn_index.num_neighbors(source);
      max_degree = std::max(m_nn_index.num_neighbors(source), max_degree);
    }
    m_comm.cf_barrier();
    if (m_option.verbose) {
      m_comm.cout0() << "#of neighbors\t"
                     << m_comm.all_reduce_sum(total_local_neighbors)
                     << std::endl;
      m_comm.cout0() << "Max #of neighbors\t"
                     << m_comm.all_reduce_max(max_degree) << std::endl;
    }
  }

  /// \warning Generated index is not sorted by distance.
  nn_index_type priv_generate_reverse_index() {
    // Only one call is allowed at a time within a process.
    static std::mutex           mutex;
    std::lock_guard<std::mutex> guard(mutex);

    nn_index_type         reversed_index(m_nn_index.get_allocator());
    static nn_index_type& ref_reversed_index(reversed_index);
    ref_reversed_index.reset();
    m_comm.cf_barrier();

    for (auto pitr = m_nn_index.points_begin(), pend = m_nn_index.points_end();
         pitr != pend; ++pitr) {
      const auto& source = pitr->first;
      for (auto nitr = m_nn_index.neighbors_begin(source),
                nend = m_nn_index.neighbors_end(source);
           nitr != nend; ++nitr) {
        const auto& neighbor = *nitr;

        // Assumes that reverse path has the same distance as the original one.
        m_comm.async(
            m_point_partitioner(neighbor.id),
            [](const id_type reverse_neighbor, const id_type reverse_source,
               const distance_type distance) {
              neighbor_type neighbor(reverse_neighbor, distance);
              ref_reversed_index.insert(reverse_source, neighbor);
            },
            source, neighbor.id, neighbor.distance);
      }
    }
    m_comm.barrier();

    return reversed_index;
  }

  void priv_prune_neighbors() {
    const std::size_t num_max_neighbors_to_retain =
        m_option.index_k * m_option.pruning_degree_multiplier;
    if (m_option.verbose) {
      m_comm.cout0() << "\nPruning neighbors"
                     << "\nEach point keeps up to "
                     << num_max_neighbors_to_retain << " neighbors"
                     << std::endl;
    }
    std::size_t count = 0;
    for (auto pitr = m_nn_index.points_begin(), pend = m_nn_index.points_end();
         pitr != pend; ++pitr) {
      const auto& source   = pitr->first;
      const auto  old_size = m_nn_index.num_neighbors(source);
      m_nn_index.prune_neighbors(source, num_max_neighbors_to_retain);
      count += old_size - m_nn_index.num_neighbors(source);
    }
    if (m_option.verbose) {
      m_comm.cout0() << "#of pruned neighbors\t" << m_comm.all_reduce_sum(count)
                     << std::endl;
    }
  }

  void priv_remove_long_paths() {
    if (m_option.verbose) {
      m_comm.cout0() << "\nRemoving long paths" << std::endl;
    }

    const std::size_t local_batch_size = m_option.batch_size / m_comm.size();

    auto pitr = m_nn_index.points_begin();
    auto pend = m_nn_index.points_end();
    while (true) {
      for (std::size_t i = 0; i < local_batch_size; ++i) {
        if (pitr == pend) {
          break;
        }
        const auto& source = pitr->first;
        if (m_nn_index.num_neighbors(source) <= 1) {
          ++pitr;
          continue;
        }
        std::vector<neighbor_type> candidates{
            m_nn_index.neighbors_begin(source) + 1,
            m_nn_index.neighbors_end(source)};
        std::sort(candidates.begin(), candidates.end());

        std::vector<id_type> retained;
        retained.push_back(candidates.front().id);
        m_comm.async(m_point_partitioner(retained.back()), long_paths_remover{},
                     m_this, source, candidates, retained);
      }
      m_comm.barrier();
      const auto finished = m_comm.all_reduce_min(pitr == pend ? 1 : 0);
      if (finished > 0) {
        break;
      }
    }
//
//    std::size_t num_retained_paths = 0;
//    std::size_t num_removed_paths  = 0;
//
//    if (m_option.verbose) {
//      m_comm.cout0() << "#of retained paths\t"
//                     << m_comm.all_reduce_sum(num_retained_paths) << std::endl;
//      m_comm.cout0() << "#of removed paths\t"
//                     << m_comm.all_reduce_sum(num_removed_paths) << std::endl;
//    }
  }


  struct long_paths_remover {
    /// \note candidates must be sorted by distance in ascending order.
    void operator()(self_pointer_type local_this, const id_type source,
                    const std::vector<neighbor_type>& candidates,
                    std::vector<id_type>              retained) {
      const auto my_id    = retained.back();
      auto&      nn_index = local_this->m_nn_index;
      assert(local_this->m_nn_index.num_neighbors(my_id) > 1);
      auto new_candidates = candidates;
      for (const auto& c : candidates) {
        bool          connect_to_candidate = false;
        distance_type me_to_candidate_distance;
        for (auto nitr = nn_index.neighbors_begin(c.id),
                  nend = nn_index.neighbors_end(c.id);
             nitr != nend; ++nitr) {
          if (nitr->id == c.id) {
            connect_to_candidate     = true;
            me_to_candidate_distance = nitr->distance;
            break;
          }
        }
        if (!connect_to_candidate) {
          continue;
        }

        const auto source_to_candidate_distance = c.distance;
        if (me_to_candidate_distance >= source_to_candidate_distance) {
          new_candidates.push_back(c);
        }
      }

      if (new_candidates.empty()) {
        // go back to source to update knn index
        local_this->m_comm.async(local_this->m_point_partitioner(source),
                                 long_paths_remover{}, local_this, source,
                                 retained);
      } else if (new_candidates.front().id == candidates.front().id) {
        retained.push_back(new_candidates.front().id);
        // keep checking
        local_this->m_comm.async(
            local_this->m_point_partitioner(retained.back()),
            long_paths_remover{}, local_this, source, new_candidates, retained);
      }
    }

    // Replace the neighbors of 'source' with 'retained'.
    void operator()(self_pointer_type local_this, const id_type source,
                    const std::vector<id_type>& retained_ids) {
      std::vector<neighbor_type> new_neighbors;
      for (const auto& id : retained_ids) {
        for (auto nitr = local_this->m_nn_index.neighbors_begin(id),
                  nend = local_this->m_nn_index.neighbors_end(id);
             nitr != nend; ++nitr) {
          if (nitr->id == id) {
            new_neighbors.push_back(*nitr);
            break;
          }
        }
      }

      auto& m_nn_index = local_this->m_nn_index;
      m_nn_index.reset_neighbors(source);
      m_nn_index.reserve_neighbors(source, new_neighbors.size());
      for (const auto n : new_neighbors) {
        m_nn_index.insert(source, n);
      }
    }
  };

  const option                               m_option;
  const point_store_type&                    m_point_store;
  const point_partitioner                    m_point_partitioner;
  const distance_metric&                     m_distance_metric;
  nn_index_type&                             m_nn_index;
  ygm::comm&                                 m_comm;
  self_pointer_type                          m_this{this};
  point_store<id_type, feature_element_type> m_point_store_cache;
};

}  // namespace saltatlas::dndetail