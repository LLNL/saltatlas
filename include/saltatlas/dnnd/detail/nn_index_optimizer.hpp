// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

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
    double      pruning_degree_multiplier{-1};  // if < 0, no pruning.
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
    auto        reversed_index = priv_generate_reverse_index();
    std::size_t max_degree     = 0;
    for (auto sitr = reversed_index.points_begin(),
              send = reversed_index.points_end();
         sitr != send; ++sitr) {
      const auto& source = sitr->first;
      assert(m_point_partitioner(source) == m_comm.rank());
      for (auto nitr = reversed_index.neighbors_begin(source),
                nend = reversed_index.neighbors_end(source);
           nitr != nend; ++nitr) {
        m_nn_index.insert(source, *nitr);
      }
      reversed_index.clear_neighbors(source);  // reduce memory usage
      m_nn_index.sort_and_remove_duplicate_neighbors(source);
      max_degree = std::max(m_nn_index.num_neighbors(source), max_degree);
    }
    m_comm.cf_barrier();
    if (m_option.verbose) {
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
    ref_reversed_index.clear();
    m_comm.cf_barrier();

    for (auto sitr = m_nn_index.points_begin(), send = m_nn_index.points_end();
         sitr != send; ++sitr) {
      const auto& source = sitr->first;
      for (auto nitr = m_nn_index.neighbors_begin(source),
                nend = m_nn_index.neighbors_end(source);
           nitr != nend; ++nitr) {
        const auto& neighbor = *nitr;

        // Assumes that reverse path has the same distance as the original one.
        m_comm.async(
            m_point_partitioner(neighbor.id),
            [](const id_type reverse_neighbor, const id_type reverse_source,
               const distance_type distance) {
              neighbor_type neighbor{.id       = reverse_neighbor,
                                     .distance = distance};
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
    for (auto sitr = m_nn_index.points_begin(), send = m_nn_index.points_end();
         sitr != send; ++sitr) {
      const auto& source   = sitr->first;
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
    // Only one call is allowed at a time within a process.
    static std::mutex           mutex;
    std::lock_guard<std::mutex> guard(mutex);

    static point_store<id_type, feature_element_type> point_store;
    m_comm.cf_barrier();

    std::size_t num_retained_paths = 0;
    std::size_t num_removed_paths  = 0;

    auto sitr = m_nn_index.points_begin();
    auto send = m_nn_index.points_end();
    while (true) {
      auto sitr2 = sitr;

      // Gather 'some' feature vectors from remote
      sitr = priv_gather_feature_vectors(sitr, send);

      // Remove long paths of points from 'sitr2' to 'sitr',
      // using the feature vectors corrected in above
      const auto counts = priv_remove_long_paths_core(sitr2, sitr);
      num_retained_paths += counts.first;
      num_removed_paths += counts.second;

      const bool finished = sitr == send;
      if (m_comm.template all_reduce_sum((int)finished) == m_comm.size()) {
        break;
      }
      point_store.clear();
    }
    if (m_option.verbose) {
      m_comm.cout0() << "#of retained paths\t"
                     << m_comm.all_reduce_sum(num_retained_paths) << std::endl;
      m_comm.cout0() << "#of removed paths\t"
                     << m_comm.all_reduce_sum(num_removed_paths) << std::endl;
    }
  }

  template <typename iterator_type>
  iterator_type priv_gather_feature_vectors(iterator_type sitr,
                                            iterator_type send) {
    std::size_t                 count = 0;
    std::unordered_set<id_type> requested;
    // Get up to 'k_batch_size' feature vectors from remote
    for (; sitr != send; ++sitr) {
      const auto& source = sitr->first;
      for (auto nitr = m_nn_index.neighbors_begin(source),
                nend = m_nn_index.neighbors_end(source);
           nitr != nend; ++nitr) {
        const auto& neighbor = *nitr;

        // Avoid requesting the same feature vector multiple times
        if (requested.count(neighbor.id)) continue;
        requested.insert(neighbor.id);

        // Get a feature vector from remote asynchronously
        m_comm.async(m_point_partitioner(neighbor.id), feature_vector_gather{},
                     m_this, neighbor.id, m_comm.rank());
      }

      if (++count > k_batch_size) break;
    }
    m_comm.barrier();

    return sitr;
  }

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
          local_this->m_remote_point_store.feature_vector(targe_id);
      target_feature.insert(target_feature.begin(), feature.begin(),
                            feature.end());
    }
  };

  template <typename iterator_type>
  std::pair<std::size_t, std::size_t> priv_remove_long_paths_core(
      iterator_type sitr, iterator_type send) {
    std::size_t num_retained_paths = 0;
    std::size_t num_removed_paths  = 0;
    for (; sitr != send; ++sitr) {
      const auto&                source = sitr->first;
      std::vector<neighbor_type> retained_neighbors;
      for (auto nitr = m_nn_index.neighbors_begin(source),
                nend = m_nn_index.neighbors_end(source);
           nitr != nend; ++nitr) {
        const auto& neighbor        = *nitr;
        const auto  distance_to_src = neighbor.distance;
        bool        remove          = false;
        for (const auto& rn : retained_neighbors) {
          const auto& rt_feature = m_remote_point_store.feature_vector(rn.id);
          const auto& n_feature =
              m_remote_point_store.feature_vector(neighbor.id);
          const auto distance_to_retained_neighbor = m_distance_metric(
              rt_feature.size(), rt_feature.data(), n_feature.data());
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
      m_nn_index.clear_neighbors(source);
      for (const auto n : retained_neighbors) {
        m_nn_index.insert(source, n);
      }
    }
    return std::make_pair(num_retained_paths, num_removed_paths);
  }

  const option            m_option;
  const point_store_type& m_point_store;
  const point_partitioner m_point_partitioner;
  const distance_metric&  m_distance_metric;
  nn_index_type&          m_nn_index;
  ygm::comm&              m_comm;
  self_pointer_type       m_this{this};
  // Use this point store to store feature vectors in remotes.
  point_store<id_type, feature_element_type> m_remote_point_store;
};

}  // namespace saltatlas::dndetail