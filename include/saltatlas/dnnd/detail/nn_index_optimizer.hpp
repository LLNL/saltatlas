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

  using featur_vector_type = typename point_store_type::feature_vector_type;
  using point_partitioner  = std::function<int(const id_type& id)>;
  using distance_metric =
      distance::metric_type<feature_element_type, distance_type>;
  using neighbor_type = typename nn_index_type::neighbor_type;

  struct option {
    std::size_t index_k{0};
    bool        undirected{false};
    double      pruning_degree_multiplier{-1};  // if <= 0, no pruning.
    bool        remove_long_paths{false};
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
    std::size_t num_retained_paths = 0;
    std::size_t num_removed_paths  = 0;

    auto pitr = m_nn_index.points_begin();
    auto pend = m_nn_index.points_end();
    // Employ a batch model to reduce communication synchronization overhead.
    const std::size_t max_batch_size = 1 << 14;
    // The batch size is determined as follows:
    // 1. Each point in the k-nn index contains 100 neighbors.
    // 2. Each feature vector has 100 dimensions.
    // 3. Each element in a feature vector is 4 bytes
    // 4. Max memory usage per process to cache feature vectors is 256 MB.
    // 100 x 100 x 4 = 40,000 bytes = 40 KB per point
    // 256 MB / 40 KB = 6,400 points per batch
    // Thus, batch_size = 2^14 should be a reasonable choice.
    const auto batch_size =
        std::min(max_batch_size, (std::size_t)std::distance(pitr, pend));
    const auto num_batches = m_comm.all_reduce_max((std::size_t)std::ceil(
        (double)std::distance(pitr, pend) / (double)batch_size));
    for (std::size_t batch_id = 0; batch_id < num_batches; ++batch_id) {
      if (m_option.verbose) {
        m_comm.cout0() << "Batch " << batch_id + 1 << "/" << num_batches
                       << std::endl;
      }

      std::vector<id_type> point_ids(batch_size);
      for (std::size_t i = 0; i < batch_size && pitr != pend; ++i) {
        point_ids[i] = pitr->first;
        ++pitr;
      }

      // Gather the feature vectors of the neighbors beforehand
      // to reduce communication overhead
      std::unordered_set<id_type> neighbor_ids;
      for (const auto& point_id : point_ids) {
        for (auto nitr = m_nn_index.neighbors_begin(point_id),
                  nend = m_nn_index.neighbors_end(point_id);
             nitr != nend; ++nitr) {
          neighbor_ids.insert(nitr->id);
        }
      }
      priv_gather_feature_vectors(neighbor_ids);

      for (const auto& point_id : point_ids) {
        const auto counts = priv_remove_long_paths_core(point_id);
        num_retained_paths += counts.first;
        num_removed_paths += counts.second;
      }
      m_point_store_cache.reset();
      m_comm.cf_barrier();
    }
    if (m_option.verbose) {
      m_comm.cout0() << "#of retained paths\t"
                     << m_comm.all_reduce_sum(num_retained_paths) << std::endl;
      m_comm.cout0() << "#of removed paths\t"
                     << m_comm.all_reduce_sum(num_removed_paths) << std::endl;
    }
  }

  /// Get the feature vectors of the neighbors of given points
  /// and stored them in 'm_point_store_cache'.
  void priv_gather_feature_vectors(const std::unordered_set<id_type>& ids) {
    m_point_store_cache.reserve(ids.size());
    m_comm.cf_barrier();
    for (const auto& id : ids) {
      // Get a feature vector from remote asynchronously
      m_comm.async(m_point_partitioner(id), feature_vector_gather{}, m_this, id,
                   m_comm.rank());
    }
    m_comm.barrier();
  }

  /// Get a feature vector from remote asynchronously and stored in
  /// 'm_point_store_cache'.
  struct feature_vector_gather {
    // First call,
    // send back the feature vector of targe_id to 'rank_to_return'.
    void operator()(self_pointer_type local_this, const id_type targe_id,
                    const int rank_to_return) {
      const auto& f = local_this->m_point_store.feature_vector(targe_id);
      const std::vector<feature_element_type> shipping_feature(f.begin(),
                                                               f.end());
      local_this->m_comm.async(rank_to_return, feature_vector_gather{},
                               local_this, targe_id, shipping_feature);
    }

    // Second call,
    // store a received feature vector.
    void operator()(self_pointer_type local_this, const id_type targe_id,
                    const std::vector<feature_element_type>& feature) {
      auto& target_feature =
          local_this->m_point_store_cache.feature_vector(targe_id);
      target_feature.insert(target_feature.begin(), feature.begin(),
                            feature.end());
    }
  };

  /// Remove long path neighbors of 'source' point.
  /// This function assumes that feature vectors of all neighbors of 'source' is
  /// stored in 'm_point_store_cache'.
  std::pair<std::size_t, std::size_t> priv_remove_long_paths_core(
      const id_type source) {
    std::size_t num_retained_paths = 0;
    std::size_t num_removed_paths  = 0;

    std::vector<neighbor_type> retained_neighbors;
    for (auto nitr = m_nn_index.neighbors_begin(source),
              nend = m_nn_index.neighbors_end(source);
         nitr != nend; ++nitr) {
      const auto& neighbor        = *nitr;
      const auto  distance_to_src = neighbor.distance;
      bool        remove          = false;
      for (const auto& rn : retained_neighbors) {
        assert(m_point_store_cache.contains(rn.id));
        const auto& rt_feature = m_point_store_cache.feature_vector(rn.id);
        const auto& n_feature = m_point_store_cache.feature_vector(neighbor.id);
        const auto  distance_to_retained_neighbor =
            m_distance_metric(rt_feature.data(), rt_feature.size(),
                              n_feature.data(), n_feature.size());
        if (distance_to_src > distance_to_retained_neighbor) {
          remove = true;
          break;
        }
      }
      if (remove) {
        ++num_removed_paths;
      } else {
        retained_neighbors.push_back(neighbor);
        ++num_retained_paths;
      }
    }

    // Update the neighbor list of 'source'
    m_nn_index.reset_neighbors(source);
    m_nn_index.reserve_neighbors(source, retained_neighbors.size());
    for (const auto n : retained_neighbors) {
      m_nn_index.insert(source, n);
    }

    return std::make_pair(num_retained_paths, num_removed_paths);
  }

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