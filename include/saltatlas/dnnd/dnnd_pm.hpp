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

#include <metall/utility/metall_mpi_adaptor.hpp>

#include <saltatlas/dnnd/detail/distance.hpp>
#include <saltatlas/dnnd/detail/dnnd_kernel.hpp>
#include <saltatlas/dnnd/detail/nn_index.hpp>
#include <saltatlas/dnnd/detail/nn_index_optimizer.hpp>
#include <saltatlas/dnnd/detail/point_store.hpp>
#include <saltatlas/dnnd/detail/query_kernel.hpp>

namespace saltatlas {

namespace dnpmdetail {
/// \brief The class the holds member variables of dnnd_pm that are stored in
/// Metall datastore.
/// \tparam Id Point ID type.
/// \tparam FeatureElement Feature Vector's element type.
/// \tparam Distance Distance type.
/// \tparam Allocator Allocator type.
template <typename Id, typename FeatureElement, typename Distance,
          typename Allocator>
struct data_core {
  using id_type              = Id;
  using feature_element_type = FeatureElement;
  using distance_type        = Distance;
  using allocator_type       = Allocator;
  using point_store_type =
      dndetail::point_store<id_type, feature_element_type, allocator_type>;
  using knn_index_type =
      dndetail::nn_index<id_type, distance_type, allocator_type>;

  data_core(const dndetail::distance::metric_id _metric_id,
            const uint64_t                      _rnd_seed,
            const allocator_type                allocator = allocator_type())
      : metric_id(_metric_id),
        rnd_seed(_rnd_seed),
        point_store(allocator),
        knn_index(allocator) {}

  dndetail::distance::metric_id metric_id;
  uint64_t                      rnd_seed;
  point_store_type              point_store;
  knn_index_type                knn_index;
  std::size_t                   index_k{0};
};
}  // namespace dnpmdetail

/// \brief Persistent Distributed NNDescent.
/// \tparam Id Point ID type.
/// \tparam FeatureElement Feature vector element type.
/// \tparam Distance Distance type.
template <typename Id = uint64_t, typename FeatureElement = double,
          typename Distance = double>
class dnnd_pm {
 private:
  using data_core_type =
      dnpmdetail::data_core<Id, FeatureElement, Distance,
                            metall::manager::allocator_type<std::byte>>;

 public:
  using id_type              = typename data_core_type::id_type;
  using feature_element_type = typename data_core_type::feature_element_type;
  using distance_type        = typename data_core_type::distance_type;
  using point_store_type     = typename data_core_type::point_store_type;
  using knn_index_type       = typename data_core_type::knn_index_type;
  using feature_vector_type  = typename point_store_type::feature_vector_type;
  using neighbor_type        = typename knn_index_type::neighbor_type;

  /// \brief Tag type to create a new graph always.
  struct create_t {};

  /// \brief Tag to create a new graph always.
  static const create_t create;

  /// \brief Tag type to open an existing graph.
  struct open_t {};

  /// \brief Tag to open an existing graph.
  static const open_t open;

  /// \brief Tag type to open an existing graph with the read only mode.
  struct open_read_only_t {};

  /// \brief Tag to open an existing graph with the read only mode.
  static const open_read_only_t open_read_only;

 private:
  using nn_kernel_type = dndetail::dnnd_kernel<point_store_type, distance_type>;
  using query_kernel_type =
      dndetail::dknn_batch_query_kernel<point_store_type, knn_index_type>;
  using nn_index_optimizer_type =
      dndetail::nn_index_optimizer<point_store_type, knn_index_type>;

 public:
  using query_point_store_type =
      typename query_kernel_type::query_point_store_type;
  using point_partitioner       = typename nn_kernel_type::point_partitioner;
  using query_result_store_type = typename query_kernel_type::knn_store_type;

  /// \brief Constructor. Create a new persistent index.
  /// \param datastore_path Path to store data.
  /// It is not allowed to contain another index at the path.
  /// \param distance_metric_name Distance metric name.
  /// \param comm YGM comm instance.
  /// \param rnd_seed Seed for random generators.
  /// \param verbose If true, enable the verbose mode.
  dnnd_pm(create_t, const std::string_view datastore_path,
          const std::string_view distance_metric_name, ygm::comm& comm,
          const uint64_t rnd_seed = std::random_device{}(),
          const bool     verbose  = false)
      : m_comm(comm), m_verbose(verbose) {
    priv_create(datastore_path, distance_metric_name, rnd_seed);
    comm.cf_barrier();
  }

  /// \brief Constructor. Opens an already stored index.
  /// \param datastore_path Path to existing index.
  /// \param comm YGM comm instance.
  /// \param verbose If true, enable the verbose mode.
  dnnd_pm(open_t, const std::string_view datastore_path, ygm::comm& comm,
          const bool verbose = false)
      : m_comm(comm), m_verbose(verbose) {
    priv_open(datastore_path);
    comm.cf_barrier();
  }

  /// \brief Constructor. Opens an already stored index with the read-only mode.
  /// \param datastore_path Path to existing index.
  /// \param comm YGM comm instance.
  /// \param verbose If true, enable the verbose mode.
  dnnd_pm(open_read_only_t, const std::string_view datastore_path,
          ygm::comm& comm, const bool verbose = false)
      : m_comm(comm), m_verbose(verbose) {
    priv_open_read_only(datastore_path);
    comm.cf_barrier();
  }

  /// \brief Return a reference to the point store instance.
  /// \return  A reference to the point store instance.
  point_store_type& get_point_store() { return m_data_core->point_store; }

  /// \brief Return a reference to the point store instance.
  /// \return  A reference to the point store instance.
  point_store_type& get_point_store() const { return m_data_core->point_store; }

  /// \brief Return a reference to a knn index instance.
  /// \return  A reference to a knn index instance.
  const knn_index_type& get_knn_index() const { return m_data_core->knn_index; }

  /// \brief Return a point partitioner instance.
  /// \return A point partitioner instance.
  point_partitioner get_point_partitioner() const {
    const int size = m_comm.size();
    return [size](const id_type& id) { return id % size; };
  };

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
    typename nn_kernel_type::option option{
        .k                          = k,
        .r                          = r,
        .delta                      = delta,
        .exchange_reverse_neighbors = exchange_reverse_neighbors,
        .mini_batch_size            = mini_batch_size,
        .rnd_seed                   = m_data_core->rnd_seed,
        .verbose                    = m_verbose};

    nn_kernel_type kernel(
        option, m_data_core->point_store, get_point_partitioner(),
        dndetail::distance::metric<feature_element_type, distance_type>(
            m_data_core->metric_id),
        m_comm);
    kernel.construct(m_data_core->knn_index);
    m_data_core->index_k = k;
  }

  /// \brief Apply some optimizations to an already constructed index aiming at
  /// improving the query quality and performance.
  /// \param make_index_undirected If true, make the index undirected.
  /// \param pruning_degree_multiplier
  /// Each point keeps up to k * pruning_degree_multiplier nearest neighbors,
  /// where k is the number of neighbors each point in the index has.
  /// if this value is less than 0, there is no pruning.
  /// \param remove_long_paths If true, remove long paths.
  void optimize_index(const bool   make_index_undirected     = false,
                      const double pruning_degree_multiplier = 1.5,
                      const bool   remove_long_paths         = false) {
    assert(m_data_core);

    const typename nn_index_optimizer_type::option opt{
        .index_k                   = m_data_core->index_k,
        .undirected                = make_index_undirected,
        .pruning_degree_multiplier = pruning_degree_multiplier,
        .remove_long_paths         = remove_long_paths,
        .verbose                   = m_verbose};
    nn_index_optimizer_type optimizer{
        opt,
        m_data_core->point_store,
        get_point_partitioner(),
        dndetail::distance::metric<feature_element_type, distance_type>(
            m_data_core->metric_id),
        m_data_core->knn_index,
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
                                              .rnd_seed = m_data_core->rnd_seed,
                                              .verbose  = m_verbose};

    query_kernel_type kernel(
        option, m_data_core->point_store, get_point_partitioner(),
        dndetail::distance::metric<feature_element_type, distance_type>(
            m_data_core->metric_id),
        m_data_core->knn_index, m_comm);

    query_result_store_type query_result;
    kernel.query_batch(query_point_store, query_result);

    return query_result;
  }

  /// \brief Takes a snapshot.
  /// \param destination_path Destination path.
  /// \return Returns true on success; otherwise, false.
  bool snapshot(const std::string_view& destination_path) {
    return m_metall->snapshot(destination_path.data());
  }

  /// \brief Checks if a datastore can be opened.
  /// \param datastore_path Datastore path.
  /// \return Returns true if the datastore is openable; otherwise, false.
  static bool openable(const std::string_view& datastore_path) {
    return metall::utility::metall_mpi_adaptor::consistent(
        datastore_path.data());
  }

  /// \brief Copies a datastore.
  /// Copying a datastore that is opened with a writable mode could cause an
  /// error. Must copy a datastore that is not opened or opened with the
  /// read-only mode.
  /// \param source_path Source data store path.
  /// \param destination_path Destination data store path.
  /// \return Returns true on success; otherwise, false.
  static bool copy(const std::string_view& source_path,
                   const std::string_view& destination_path) {
    return metall::utility::metall_mpi_adaptor::copy(source_path.data(),
                                                     destination_path.data());
  }

  /// \brief Removes a datastore.
  /// \param datastore_path Datastore path.
  /// \return Returns true on success; otherwise, false.
  static bool remove(const std::string_view& datastore_path) {
    return metall::utility::metall_mpi_adaptor::remove(datastore_path.data());
  }

 private:
  void priv_create_metall(const std::string_view path) {
    m_metall = std::make_unique<metall::utility::metall_mpi_adaptor>(
        metall::create_only, path.data(), MPI_COMM_WORLD);
  }

  void priv_create(const std::string_view path,
                   const std::string_view distance_metric_name,
                   const uint64_t         rnd_seed) {
    priv_create_metall(path);
    auto& lmgr  = m_metall->get_local_manager();
    m_data_core = lmgr.construct<data_core_type>(metall::unique_instance)(
        dndetail::distance::convert_to_metric_id(distance_metric_name),
        rnd_seed, lmgr.get_allocator());
    assert(m_data_core);
  }

  void priv_open_metall(const std::string_view path) {
    m_metall = std::make_unique<metall::utility::metall_mpi_adaptor>(
        metall::open_only, path.data(), MPI_COMM_WORLD);
  }

  void priv_open(const std::string_view datastore_path) {
    priv_open_metall(datastore_path);
    auto& local_manager = m_metall->get_local_manager();
    m_data_core =
        local_manager.find<data_core_type>(metall::unique_instance).first;
    assert(m_data_core);
  }

  void priv_open_read_only_metall(const std::string_view path) {
    m_metall = std::make_unique<metall::utility::metall_mpi_adaptor>(
        metall::open_read_only, path.data(), MPI_COMM_WORLD);
  }

  void priv_open_read_only(const std::string_view datastore_path) {
    priv_open_read_only_metall(datastore_path);
    const auto& local_manager = m_metall->get_local_manager();
    m_data_core =
        local_manager.find<data_core_type>(metall::unique_instance).first;
    assert(m_data_core);
  }

  ygm::comm&                                           m_comm;
  std::unique_ptr<metall::utility::metall_mpi_adaptor> m_metall{nullptr};
  data_core_type*                                      m_data_core{nullptr};
  bool                                                 m_verbose{false};
};

}  // namespace saltatlas
