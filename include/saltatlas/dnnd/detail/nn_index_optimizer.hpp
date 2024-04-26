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
#include "saltatlas/point_store.hpp"

namespace saltatlas::dndetail {

template <typename PointStore, typename KNNIndex>
class nn_index_optimizer {
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
      saltatlas::distance::distance_function_type<point_type, distance_type>;
  using neighbor_type = typename nn_index_type::neighbor_type;

  struct option {
    std::size_t index_k{0};
    bool        undirected{false};
    double      pruning_degree_multiplier{-1};  // if <= 0, no pruning.
    bool        remove_long_paths{false};
    std::size_t batch_size{1 << 25};  // if <= 0, process all at a time.
    bool        verbose{false};
  };

  nn_index_optimizer(const option& opt, const point_store_type& point_store,
                     const point_partitioner&      partitioner,
                     const distance_function_type& dist_function,
                     nn_index_type& nn_index, ygm::comm& comm)
      : m_option(opt),
        m_point_store(point_store),
        m_point_partitioner(partitioner),
        m_distance_function(dist_function),
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
    if (m_option.verbose) {
      m_comm.cout0() << "#of original neighbors\t"
                     << m_comm.all_reduce_sum(m_nn_index.count_all_neighbors())
                     << std::endl;
    }

    auto        reversed_index = priv_generate_reverse_index();
    std::size_t max_degree     = 0;
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
      max_degree = std::max(m_nn_index.num_neighbors(source), max_degree);
    }
    m_comm.cf_barrier();
    if (m_option.verbose) {
      m_comm.cout0() << "#of neighbors\t"
                     << m_comm.all_reduce_sum(m_nn_index.count_all_neighbors())
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
      m_comm.cout0() << "\nPruning neighbors" << "\nEach point keeps up to "
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

    const auto global_initial_num_neighbors =
        m_comm.all_reduce_sum(m_nn_index.count_all_neighbors());

    const std::size_t local_batch_size =
        (m_option.batch_size > 0) ? m_option.batch_size / m_comm.size()
                                  : std::numeric_limits<std::size_t>::max();

    auto        pitr     = m_nn_index.points_begin();
    const auto  pend     = m_nn_index.points_end();
    std::size_t batch_no = 0;
    while (true) {
      if (m_option.verbose) {
        m_comm.cout0() << "Batch #" << batch_no << std::endl;
      }
      for (std::size_t i = 0; i < local_batch_size && pitr != pend;
           ++i, ++pitr) {
        const auto& source = pitr->first;
        if (m_nn_index.num_neighbors(source) <= 1) {
          continue;
        }

        std::vector<neighbor_type> candidates{
            m_nn_index.neighbors_begin(source),
            m_nn_index.neighbors_end(source)};
        std::sort(candidates.begin(), candidates.end());

        // Closest neighbor is always retained.
        std::vector<id_type> retained_neighbors;
        retained_neighbors.push_back(candidates.front().id);
        candidates.erase(candidates.begin());

        m_comm.async(m_point_partitioner(retained_neighbors.back()),
                     distance_based_path_selector{}, m_this, source, candidates,
                     retained_neighbors);
      }
      m_comm.barrier();
      const auto finished =
          m_comm.all_reduce_min(std::size_t(pitr == pend ? 1 : 0));
      if (finished > 0) {
        break;
      }
      ++batch_no;
    }

    const auto global_retained_num_neighbors =
        m_comm.all_reduce_sum(m_nn_index.count_all_neighbors());
    const auto global_removed_num_neighbors =
        global_initial_num_neighbors - global_retained_num_neighbors;
    if (m_option.verbose) {
      m_comm.cout0() << "#of removed neighbors\t"
                     << global_removed_num_neighbors << std::endl;
      m_comm.cout0() << "#of retained neighbors\t"
                     << global_retained_num_neighbors << std::endl;
      m_comm.cout0() << "Removed ratio\t"
                     << (double(global_removed_num_neighbors) /
                         global_initial_num_neighbors)
                     << std::endl;
    }
  }

  /// \brief Selects neighbors worth to be retained based on distance.
  struct distance_based_path_selector {
    /// \note candidates must be sorted by distance in ascending order.
    void operator()(self_pointer_type local_this, const id_type source,
                    const std::vector<neighbor_type>& candidates,
                    std::vector<id_type>              retained) {
      assert(!retained.empty());
      const auto my_id    = retained.back();
      auto&      nn_index = local_this->m_nn_index;
      assert(local_this->m_nn_index.num_neighbors(my_id) > 0);

      std::vector<neighbor_type> new_candidates;
      for (std::size_t i = 0; i < candidates.size(); ++i) {
        auto& candidate = candidates[i];
        assert(candidate.id != my_id);

        // Check if I connect to the candidate.
        bool          connect_to_candidate = false;
        distance_type me_to_candidate_distance;
        for (auto nitr = nn_index.neighbors_begin(my_id),
                  nend = nn_index.neighbors_end(my_id);
             nitr != nend; ++nitr) {
          if (nitr->id == candidate.id) {
            connect_to_candidate     = true;
            me_to_candidate_distance = nitr->distance;
            break;
          }
        }

        const auto source_to_candidate_distance = candidate.distance;
        if (connect_to_candidate &&
            me_to_candidate_distance >= source_to_candidate_distance) {
          // The candidate is not worth to keep.
          // Does not include in the new candidate list.
          continue;
        }

        if (new_candidates.empty()) {
          // As the candidate list is sorted by the distance from the source,
          // the first candidate that passes the dropping test is allowed to
          // include to the retained neighbors list.
          retained.push_back(candidate.id);
        } else {
          // Still not sure if it is worth keeping the candidate.
          new_candidates.push_back(candidate);
        }
      }

      if (new_candidates.empty()) {
        // go back to the source to update knn index as there is no more
        // candidates to check.
        local_this->m_comm.async(local_this->m_point_partitioner(source),
                                 distance_based_path_selector{}, local_this,
                                 source, retained);
      } else {
        // keep the selection
        local_this->m_comm.async(
            local_this->m_point_partitioner(retained.back()),
            distance_based_path_selector{}, local_this, source, new_candidates,
            retained);
      }
    }

    // Replace the neighbors of 'source' with the retained ones.
    void operator()(self_pointer_type local_this, const id_type source,
                    const std::vector<id_type>& retained_ids) {
      assert(!retained_ids.empty());
      auto& nn_index = local_this->m_nn_index;

      std::vector<neighbor_type> new_neighbors;
      for (const auto& id : retained_ids) {
        for (auto nitr = nn_index.neighbors_begin(source),
                  nend = nn_index.neighbors_end(source);
             nitr != nend; ++nitr) {
          if (nitr->id == id) {
            new_neighbors.push_back(*nitr);
            break;
          }
        }
      }
      assert(!new_neighbors.empty());

      nn_index.reset_neighbors(source);
      nn_index.reserve_neighbors(source, new_neighbors.size());
      for (const auto n : new_neighbors) {
        nn_index.insert(source, n);
      }
    }
  };

  const option                  m_option;
  const point_store_type&       m_point_store;
  const point_partitioner       m_point_partitioner;
  const distance_function_type& m_distance_function;
  nn_index_type&                m_nn_index;
  ygm::comm&                    m_comm;
  self_pointer_type             m_this{this};
};

}  // namespace saltatlas::dndetail