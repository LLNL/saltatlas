// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <ygm/comm.hpp>
#include <ygm/utility.hpp>

#include <saltatlas/dnnd/detail/distance.hpp>
#include <saltatlas/dnnd/detail/dnnd_kernel.hpp>
#include <saltatlas/dnnd/detail/nn_index.hpp>
#include <saltatlas/dnnd/detail/nn_index_optimizer.hpp>
#include <saltatlas/dnnd/detail/query_kernel.hpp>
#include <saltatlas/dnnd/feature_vector.hpp>
#include "saltatlas/point_store.hpp"

// Ideas:
// Do not use data_core anymore?
// Does not have to keep distance_id
// Use base_dnnd as a driver (only static functions)?

namespace saltatlas::dndetail {

/// \brief The class the holds member variables of base_dnnd class.
/// \tparam Id Point ID type.
/// \tparam Point Point type.
/// \tparam Distance Distance type.
/// \tparam Allocator Allocator type.
template <typename Id, typename Point, typename Distance, typename Allocator>
struct data_core {
  using id_type          = Id;
  using point_type       = Point;
  using distance_type    = Distance;
  using allocator_type   = Allocator;
  using point_store_type = point_store<id_type, point_type, std::hash<id_type>,
                                       std::equal_to<>, allocator_type>;
  using knn_index_type =
      dndetail::nn_index<id_type, distance_type, allocator_type>;

  data_core(const saltatlas::distance::id _distance_id,
            const uint64_t                _rnd_seed,
            const allocator_type          allocator = allocator_type())
      : distance_id(_distance_id),
        rnd_seed(_rnd_seed),
        pstore(allocator),
        knn_index(allocator) {}

  saltatlas::distance::id distance_id;
  uint64_t                rnd_seed;  // TODO: does not have to hold?
  point_store_type        pstore;
  knn_index_type          knn_index;
  std::size_t             index_k{0};
};

/// \brief Base class of the Distributed NNDescent.
/// \tparam Id Point ID type.
/// \tparam PointType Point type.
/// \tparam Distance Distance type.
template <typename Id = uint64_t, typename PointType = feature_vector<double>,
          typename Distance  = double,
          typename Allocator = std::allocator<std::byte>>
class base_dnnd {
 private:
  using self_type = base_dnnd<Id, PointType, Distance, Allocator>;

 public:
  using data_core_type = data_core<Id, PointType, Distance, Allocator>;

  using id_type          = typename data_core_type::id_type;
  using point_type       = typename data_core_type::point_type;
  using distance_type    = typename data_core_type::distance_type;
  using point_store_type = typename data_core_type::point_store_type;
  using knn_index_type   = typename data_core_type::knn_index_type;
  using neighbor_type    = typename knn_index_type::neighbor_type;
  using distance_function_type =
      distance::distance_function_type<point_type, distance_type>;

 private:
  using nn_kernel_type = dndetail::dnnd_kernel<point_store_type, distance_type>;
  using query_kernel_type =
      dndetail::dknn_batch_query_kernel<point_store_type, knn_index_type>;
  using nn_index_optimizer_type =
      dndetail::nn_index_optimizer<point_store_type, knn_index_type>;

 public:
  using query_store_type    = typename query_kernel_type::query_store_type;
  using point_partitioner   = typename nn_kernel_type::point_partitioner;
  using neighbor_store_type = typename query_kernel_type::neighbor_store_type;

  base_dnnd(const bool verbose, ygm::comm& comm)
      : m_data_core(), m_comm(comm), m_verbose(verbose) {
    m_comm.cf_barrier();
  }

  ~base_dnnd() noexcept = default;

  base_dnnd(const base_dnnd&)                = delete;
  base_dnnd& operator=(const base_dnnd&)     = delete;
  base_dnnd(base_dnnd&&) noexcept            = default;
  base_dnnd& operator=(base_dnnd&&) noexcept = default;

  /// \brief Return a reference to the point store instance.
  /// \return  A reference to the point store instance.
  point_store_type& get_point_store() { return m_data_core->pstore; }

  /// \brief Return a reference to the point store instance.
  /// \return  A reference to the point store instance.
  const point_store_type& get_point_store() const {
    return m_data_core->pstore;
  }

  /// \brief Return a reference to a knn index instance.
  /// \return  A reference to a knn index instance.
  const knn_index_type& get_knn_index() const { return m_data_core->knn_index; }

  /// \brief Return a point partitioner instance.
  /// \return A point partitioner instance.
  point_partitioner get_point_partitioner() const {
    const int size = m_comm.size();
    // TODO: hash id?
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
    auto kernel = priv_init_kernel(k, r, delta, exchange_reverse_neighbors,
                                   mini_batch_size);
    kernel.construct(m_data_core->knn_index);
    m_data_core->index_k = k;
  }

  /// \brief Construct an k-NN index.
  /// Take a neighbor data for initialization.
  /// \param k The number of nearest neighbors each point in the index has.
  /// \param r Sample rate parameter in NN-Descent.
  /// \param delta Precision parameter in NN-Descent.
  /// \param exchange_reverse_neighbors If true is specified, exchange reverse
  /// neighbors globally.
  /// \param mini_batch_size Mini batch size.
  /// \param init_neighbors Neighbor data for initialization.
  /// \param settled_init_index If true, treat the initial neighbors as 'old'
  /// ones.
  void construct_index(
      const int k, const double r, const double delta,
      const bool exchange_reverse_neighbors, const std::size_t mini_batch_size,
      const std::unordered_map<id_type, std::vector<id_type>>& init_neighbors,
      bool settled_init_index) {
    auto kernel = priv_init_kernel(k, r, delta, exchange_reverse_neighbors,
                                   mini_batch_size);
    kernel.construct(init_neighbors, settled_init_index,
                     m_data_core->knn_index);
    m_data_core->index_k = k;
  }

  /// \brief Construct an k-NN index.
  /// Take an existing k-NN index for initialization.
  /// \tparam init_index_alloc_type Allocator type for init_index.
  /// \param k The number of nearest neighbors each point in the index has.
  /// \param r Sample rate parameter in NN-Descent.
  /// \param delta Precision parameter in NN-Descent.
  /// \param exchange_reverse_neighbors If true is specified, exchange reverse
  /// neighbors globally.
  /// \param mini_batch_size Mini batch size.
  /// \param init_index k-NN index for initialization.
  /// \param settled_init_index If true, treat the initial neighbors as 'old'
  /// ones.
  template <typename init_index_alloc_type>
  void construct_index(
      const int k, const double r, const double delta,
      const bool exchange_reverse_neighbors, const std::size_t mini_batch_size,
      const nn_index<id_type, distance_type, init_index_alloc_type>& init_index,
      bool settled_init_index) {
    auto kernel = priv_init_kernel(k, r, delta, exchange_reverse_neighbors,
                                   mini_batch_size);
    kernel.construct(init_index, settled_init_index, m_data_core->knn_index);
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
    nn_index_optimizer_type optimizer{opt,
                                      m_data_core->pstore,
                                      get_point_partitioner(),
                                      m_distance_function,
                                      m_data_core->knn_index,
                                      m_comm};
    optimizer.run();
  }

  /// \brief Query nearest neighbors of given points.
  /// \param queries Queries (a list of the feature vectors of query points).
  /// Assume that queries are already partitioned.
  /// \param k The number of nearest neighbors to search for each point.
  /// \param batch_size The number of queries to process at a time globally.
  /// \return Computed k nearest neighbors of the given points.
  /// Returned as an adjacency list (vector of vectors).
  /// Specifically, k nearest neighbor data of the i-th query is stored in the
  /// i-th inner vector. Each inner vector contains pairs of a neighbor ID and a
  /// distance to the neighbor from the query point.
  neighbor_store_type query_batch(const std::vector<point_type>& queries,
                                  const int k, const double epsilon,
                                  const double      mu,
                                  const std::size_t batch_size) {
    typename query_kernel_type::option option{.k          = k,
                                              .epsilon    = epsilon,
                                              .mu         = mu,
                                              .batch_size = batch_size,
                                              .rnd_seed = m_data_core->rnd_seed,
                                              .verbose  = m_verbose};

    query_kernel_type kernel(option, m_data_core->pstore,
                             get_point_partitioner(), m_distance_function,
                             m_data_core->knn_index, m_comm);

    neighbor_store_type query_result;
    kernel.query_batch(queries, query_result);

    return query_result;
  }

  /// \brief Dump the k-NN index to distributed files.
  /// \param out_file_prefix File path prefix.
  /// \param dump_distance If true, also dump distances
  /// \details For each neighbor list, the following lines are dumped:
  /// ```
  /// source_id neighbor_id_1 neighbor_id_2 ...
  /// 0.0 distance_1 distance_2 ...
  /// ```
  /// Each item is separated by a tab. The first line is the source id and
  /// neighbor ids. The second line is the dummy value and distances to each
  /// neighbor. The dummy value is just a placeholder so that each neighbor id
  /// and distance pair is stored in the same column.
  bool dump_index(const std::string_view out_file_prefix,
                  const bool             dump_distance = false) const {
    return priv_dump_index_distributed_file(out_file_prefix, dump_distance);
  }

  /// \brief Returns the distance metric name.
  std::string get_distance_name() const {
    return distance::convert_to_distance_name(m_data_core->distance_id);
  }

  /// \brief Returns YGM communicator.
  ygm::comm& get_comm() const { return m_comm; }

 private:
  /// \brief Initialize the internal data core instance.
  /// \param data_core A data core instance.
  void priv_set_data_core(data_core_type&               data_core,
                          const distance_function_type& distance_function) {
    if (m_data_core) {
      m_comm.cerr0() << "Data core is already initialized." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    m_data_core = &data_core;

    m_distance_function = distance_function;
  }

  /// \brief set_data_core for using a pre-defined distance function.
  /// The data core argument must contain a valid distance function ID.
  void priv_set_data_core(data_core_type& data_core) {
    priv_set_data_core(data_core,
                       distance::distance_function<point_type, distance_type>(
                           data_core.distance_id));
  }

  nn_kernel_type priv_init_kernel(const int k, const double r,
                                  const double      delta,
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

    nn_kernel_type kernel(option, m_data_core->pstore, get_point_partitioner(),
                          m_distance_function, m_comm);

    return kernel;
  }

  bool priv_dump_index_distributed_file(const std::string_view out_file_prefix,
                                        const bool dump_distance) const {
    std::stringstream file_name;
    file_name << out_file_prefix << "-" << m_comm.rank();
    const auto ret =
        m_data_core->knn_index.dump(file_name.str(), dump_distance);
    return ret;
  }

  data_core_type* m_data_core;
  ygm::comm*      m_comm;
  bool m_verbose{false};  // TODO: make this changeable after construction
  distance_function_type m_distance_function;
};

}  // namespace saltatlas::dndetail