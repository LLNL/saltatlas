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
  using distance_metric = saltatlas::distance::metric_type<featur_vector_type>;
  using neighbor_type   = typename nn_index_type::neighbor_type;

  struct option {
    std::size_t index_k{0};
    bool        undirected{false};
    double      pruning_degree_multiplier{1.5};
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
        m_comm(comm),
        m_tmp_index(m_nn_index.get_allocator()) {
    m_this.check(m_comm);
  }

  void run() {
    if (m_option.undirected) {
      priv_make_index_undirected();
    }
    if (!nearly_equal(m_option.pruning_degree_multiplier, 1.0)) {
      priv_prune_neighbors();
    }
    if (m_option.remove_long_paths) {
      priv_remove_long_paths();
    }
    m_comm.cf_barrier();
  }

 private:
  using self_type         = nn_index_optimizer<PointStore, KNNIndex>;
  using self_pointer_type = ygm::ygm_ptr<self_type>;

  void priv_make_index_undirected() {
    auto reversed_index = priv_generate_reverse_index();
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
    }
    m_comm.cf_barrier();
  }

  /// \warning Generated index is not sorted by distance.
  nn_index_type priv_generate_reverse_index() {
    m_tmp_index.clear();
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
            [](self_pointer_type local_this, const id_type reverse_neighbor,
               const id_type reverse_source, const distance_type distance) {
              neighbor_type neighbor{.id       = reverse_neighbor,
                                     .distance = distance};
              local_this->m_tmp_index.insert(reverse_source, neighbor);
            },
            m_this, source, neighbor.id, neighbor.distance);
      }
    }
    m_comm.barrier();
    return m_tmp_index;
  }

  void priv_prune_neighbors() {
    const std::size_t num_max_neighbors_to_retain =
        m_option.index_k * m_option.pruning_degree_multiplier;
    for (auto sitr = m_nn_index.points_begin(), send = m_nn_index.points_end();
         sitr != send; ++sitr) {
      const auto& source = sitr->first;
      m_nn_index.prune_neighbors(source, num_max_neighbors_to_retain);
    }
    m_comm.cf_barrier();
  }

  void priv_remove_long_paths() {
    // TODO: implement later.
    assert(false);
  }

  const option            m_option;
  const point_store_type& m_point_store;
  const point_partitioner m_point_partitioner;
  const distance_metric&  m_distance_metric;
  nn_index_type&          m_nn_index;
  ygm::comm&              m_comm;
  nn_index_type           m_tmp_index;
  self_pointer_type       m_this{this};
};

}  // namespace saltatlas::dndetail