// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/detail/dnnd_kernel.hpp>
#include <saltatlas/dnnd/detail/nn_index.hpp>
#include <saltatlas/dnnd/detail/nn_index_optimizer.hpp>
#include <saltatlas/dnnd/detail/point_store.hpp>
#include <saltatlas/dnnd/detail/query_kernel.hpp>
#include <saltatlas/dnnd/distance.hpp>
#include "point_reader.hpp"

namespace saltatlas {

template <typename Id = uint64_t, typename FeatureElement = double,
          typename Distance = double>
class dnnd {
 private:
 public:
  using id_type              = Id;
  using feature_element_type = FeatureElement;
  using distance_type        = Distance;
  using point_store_type = dndetail::point_store<id_type, feature_element_type>;
  using knn_index_type   = dndetail::nn_index<id_type, distance_type>;
  using feature_vector_type = typename point_store_type::feature_vector_type;
  using neighbor_type       = typename knn_index_type::neighbor_type;

 private:
  using nn_kernel_type = dndetail::dnnd_kernel<point_store_type, distance_type>;
  using query_kernel_type =
      dndetail::dknn_batch_query_kernel<point_store_type, knn_index_type>;
  using nn_index_optimizer_type =
      dndetail::nn_index_optimizer<point_store_type, knn_index_type>;

 public:
  using query_point_store_type =
      typename query_kernel_type::query_point_store_type;
  using query_result_store_type = typename query_kernel_type::knn_store_type;

  dnnd(const std::string_view          distance_metric_name,
       const std::vector<std::string>& point_file_names,
       const std::string_view point_file_format, ygm::comm& comm,
       const uint64_t rnd_seed = std::random_device{}(),
       const bool     verbose  = false)
      : m_distance_metric(dndetail::distance::metric<feature_vector_type>(
            distance_metric_name)),
        m_point_file_names(point_file_names),
        m_point_file_format(point_file_format),
        m_comm(comm),
        m_rnd_seed(rnd_seed),
        m_verbose(verbose) {}

  /// \brief Construct an k-NN index.
  /// \param k The number of nearest neighbors each point in the index has.
  /// \param r Sample rate parameter in NN-Descent.
  /// \param delta Precision parameter in NN-Descent.
  /// \param exchange_reverse_neighbors If true is specified, exchange reverse
  /// neighbors globally.
  /// \param mini_batch_size Mini batch size.
  void construct_index(const int k, const double r, const double delta,
                       const bool        exchange_reverse_neighbors,
                       const std::size_t mini_batch_size) {
    priv_read_points();

    typename nn_kernel_type::option option{
        .k                          = k,
        .r                          = r,
        .delta                      = delta,
        .exchange_reverse_neighbors = exchange_reverse_neighbors,
        .mini_batch_size            = mini_batch_size,
        .rnd_seed                   = m_rnd_seed,
        .verbose                    = m_verbose};

    nn_kernel_type kernel(option, *m_point_store, priv_get_point_partitioner(),
                          m_distance_metric, m_comm);
    kernel.construct(*m_knn_index);
    m_index_k = k;
  }

  /// \brief Apply some optimizations to an already constructed index aiming at
  /// improving the query quality and performance.
  /// \param make_index_undirected If true, make the index undirected.
  /// \param pruning_degree_multiplier
  /// Each point keeps up to k * pruning_degree_multiplier nearest neighbors,
  /// where k is the number of neighbors each point in the index has.
  /// \param remove_long_paths If true, remove long paths.
  void optimize_index(const bool   make_index_undirected     = false,
                      const double pruning_degree_multiplier = 1.5,
                      const bool   remove_long_paths         = false) {
    if (!m_knn_index) {
      m_comm.cerr0() << "The source index is empty." << std::endl;
      return;
    }

    const typename nn_index_optimizer_type::option opt{
        .index_k                   = m_index_k,
        .undirected                = make_index_undirected,
        .pruning_degree_multiplier = pruning_degree_multiplier,
        .remove_long_paths         = remove_long_paths,
        .verbose                   = m_verbose};
    nn_index_optimizer_type optimizer{opt,
                                      *m_point_store,
                                      priv_get_point_partitioner(),
                                      m_distance_metric,
                                      *m_knn_index,
                                      m_comm};
    optimizer.run();
  }

  /// \brief Query nearest neighbors of given points.
  /// \param query_point_store Query points. All ranks have to have the same
  /// points.
  /// \param k The number of neighbors to search for each point.
  /// \param batch_size The number of queries to process at a time.
  /// \return Query results.
  query_result_store_type query_batch(
      const query_point_store_type& query_point_store, const int k,
      const std::size_t batch_size) {
    typename query_kernel_type::option option{.k          = k,
                                              .batch_size = batch_size,
                                              .rnd_seed   = m_rnd_seed,
                                              .verbose    = m_verbose};

    query_kernel_type kernel(option, *m_point_store,
                             priv_get_point_partitioner(), m_distance_metric,
                             *m_knn_index, m_comm);

    query_result_store_type query_result;
    kernel.query_batch(query_point_store, query_result);

    return query_result;
  }

  /// \brief Dump the k-NN index to files.
  /// \param out_file_prefix File path prefix.
  void dump_index(const std::string& out_file_prefix) {
    priv_dump_index_distributed_file(*m_knn_index, out_file_prefix);
  }

 private:
  auto priv_get_point_partitioner() const {
    const int size = m_comm.size();
    return [size](const id_type& id) { return id % size; };
  }

  void priv_read_points() {
    priv_allocate();
    m_point_store->clear();
    read_points(m_point_file_names, m_point_file_format, m_verbose,
                *m_point_store, m_comm);
  }

  void priv_allocate() {
    if (!m_point_store) {
      m_point_store = std::make_unique<point_store_type>();
    }
    if (!m_knn_index) {
      m_knn_index = std::make_unique<knn_index_type>();
    }
  }

  void priv_dump_index_distributed_file(const knn_index_type& knn_index,
                                        const std::string&    out_file_prefix) {
    std::stringstream file_name;
    file_name << out_file_prefix << "-" << m_comm.rank();
    std::ofstream ofs(file_name.str());
    if (!ofs.is_open()) {
      std::cerr << "Failed to create: " << file_name.str() << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (auto sitr = knn_index.points_begin(), send = knn_index.points_end();
         sitr != send; ++sitr) {
      const auto& source = sitr->first;
      ofs << source;
      for (auto nitr = knn_index.neighbors_begin(source),
                nend = knn_index.neighbors_end(source);
           nitr != nend; ++nitr) {
        ofs << "\t" << nitr->id;
      }
      ofs << "\n";
      ofs << "0.0";
      for (auto nitr = knn_index.neighbors_begin(source),
                nend = knn_index.neighbors_end(source);
           nitr != nend; ++nitr) {
        ofs << "\t" << nitr->distance;
      }
      ofs << "\n";
    }
    ofs.close();
    if (!ofs) {
      std::cerr << "Failed to write data to: " << file_name.str() << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    m_comm.cf_barrier();
  }

  const dndetail::distance::metric_type<feature_vector_type>& m_distance_metric;
  const std::vector<std::string>    m_point_file_names;
  const std::string                 m_point_file_format;
  ygm::comm&                        m_comm;
  uint64_t                          m_rnd_seed;
  bool                              m_verbose;
  std::unique_ptr<point_store_type> m_point_store;
  std::unique_ptr<knn_index_type>   m_knn_index;
  std::size_t                       m_index_k{0};
};

}  // namespace saltatlas
