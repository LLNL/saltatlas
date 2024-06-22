// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <filesystem>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <string_view>

#include <metall/container/vector.hpp>
#include <metall/metall.hpp>
#include <ygm/comm.hpp>

#include "saltatlas/dnnd/data_reader.hpp"
#include "saltatlas/dnnd/detail/distance.hpp"
#include "saltatlas/dnnd/detail/dnnd_kernel.hpp"
#include "saltatlas/dnnd/detail/nn_index.hpp"
#include "saltatlas/dnnd/detail/nn_index_optimizer.hpp"
#include "saltatlas/dnnd/detail/query_kernel.hpp"
#include "saltatlas/dnnd/detail/utilities/hash.hpp"
#include "saltatlas/dnnd/detail/utilities/iterator_proxy.hpp"
#include "saltatlas/dnnd/feature_vector.hpp"
#include "saltatlas/point_store.hpp"

namespace saltatlas {

/// \brief Distributed NNDescent simple version.
/// \tparam Id Point ID type.
/// \tparam Point Point type.
/// \tparam Distance Distance type.
template <typename Id       = uint64_t,
          typename Point    = saltatlas::feature_vector<double>,
          typename Distance = double>
class dnnd {
 public:
  /// \brief Point ID type.
  using id_type = Id;
  /// \brief Distance type.
  using distance_type = Distance;
  /// \brief Point type.
  using point_type = Point;

 private:
  template <typename T>
  using allocator_type = metall::manager::fallback_allocator<T>;

  template <typename T>
  using scp_allocator_type = metall::manager::scoped_fallback_allocator_type<T>;

  /// \brief Point store type.
  using point_store_type =
      point_store<id_type, point_type, std::hash<id_type>, std::equal_to<>,
                  allocator_type<std::byte>>;

  /// \brief k-NN index type.
  using knn_index_type =
      dndetail::nn_index<id_type, distance_type, allocator_type<std::byte>>;

  using nn_kernel_type = dndetail::dnnd_kernel<point_store_type, distance_type>;

  /// \brief Point partitioner type.
  using point_partitioner = typename nn_kernel_type::point_partitioner;

  using nn_index_optimizer_type =
      dndetail::nn_index_optimizer<point_store_type, knn_index_type>;

  using query_kernel_type =
      dndetail::dknn_batch_query_kernel<point_store_type, knn_index_type>;

  using query_store_type = typename query_kernel_type::query_store_type;

  using knn_index_container =
      metall::container::vector<knn_index_type,
                                scp_allocator_type<knn_index_type>>;
  using size_container =
      metall::container::vector<std::size_t, scp_allocator_type<std::size_t>>;

 public:
  /// \brief Neighbor type (contains a neighbor ID and the distance to the
  /// neighbor).
  using neighbor_type = typename knn_index_type::neighbor_type;

  using iterator_proxy_type =
      dndetail::iterator_proxy<typename point_store_type::const_iterator>;

  /// \brief Distance function type.
  /// Specifically, std::function<distance_type(const point_type &, const
  /// point_type &)>.
  using distance_function_type =
      distance::distance_function_type<point_type, distance_type>;

  /// \brief Query result store type. Specifically,
  /// std::vector<std::vector<neighbor_type>>.
  using neighbor_store_type = typename query_kernel_type::neighbor_store_type;

  /// \brief Tag type to create the Metall datastore always.
  /// The existing Metall datastore with the same name is over written.
  struct create_only_t {};

  /// \brief Tag to create the Metall datastore always.
  /// The existing Metall datastore with the same name is over written.
  [[maybe_unused]] static constexpr create_only_t create_only{};

  /// \brief Tag type to open an already created Metall datastore.
  struct open_only_t {};

  /// \brief Tag to open an already created Metall datastore.
  [[maybe_unused]] static constexpr open_only_t open_only{};

  /// \brief Tag type to open an already created Metall datastore as read only.
  struct open_read_only_t {};

  /// \brief Tag to open an already created segment as read only.
  [[maybe_unused]] static constexpr open_read_only_t open_read_only{};

  /// \brief Constructor.
  /// \param comm YGM comm instance.
  /// \param rnd_seed Seed for random generators.
  /// \param verbose If true, enable the verbose mode.
  explicit dnnd(ygm::comm&     comm,
                const uint64_t rnd_seed = std::random_device{}(),
                const bool     verbose  = false)
      : m_comm(comm), m_rnd_seed(rnd_seed), m_verbose(verbose) {
    m_pstore         = new point_store_type();
    m_knn_index_list = new knn_index_container();
    m_index_k_list   = new size_container();
  }

  dnnd(create_only_t, const std::filesystem::path& datastore_path,
       ygm::comm& comm, const uint64_t rnd_seed = std::random_device{}(),
       const bool verbose = false)
      : m_comm(comm), m_rnd_seed(rnd_seed), m_verbose(verbose) {
    m_metall_manager = std::make_unique<metall::manager>(
        metall::create_only, datastore_path.string());
    m_pstore = m_metall_manager->construct<point_store_type>(
        metall::unique_instance)(m_metall_manager->get_allocator<>());
    m_knn_index_list = m_metall_manager->construct<knn_index_container>(
        metall::unique_instance)(m_metall_manager->get_allocator<>());
    m_index_k_list = m_metall_manager->construct<size_container>(
        metall::unique_instance)(m_metall_manager->get_allocator<>());
  }

  dnnd(open_only_t, const std::filesystem::path& datastore_path,
       ygm::comm& comm, const uint64_t rnd_seed = std::random_device{}(),
       const bool verbose = false)
      : m_comm(comm), m_rnd_seed(rnd_seed), m_verbose(verbose) {
    m_metall_manager = std::make_unique<metall::manager>(
        metall::open_only, datastore_path.string());
    m_pstore =
        m_metall_manager->find<point_store_type>(metall::unique_instance).first;
    m_knn_index_list =
        m_metall_manager->find<knn_index_container>(metall::unique_instance)
            .first;
    m_index_k_list =
        m_metall_manager->find<size_container>(metall::unique_instance).first;
  }

  dnnd(open_read_only_t, const std::filesystem::path& datastore_path,
       ygm::comm& comm, const uint64_t rnd_seed = std::random_device{}(),
       const bool verbose = false)
      : m_comm(comm), m_rnd_seed(rnd_seed), m_verbose(verbose) {
    m_metall_manager = std::make_unique<metall::manager>(
        metall::open_read_only, datastore_path.string());
    m_pstore =
        m_metall_manager->find<point_store_type>(metall::unique_instance).first;
    m_knn_index_list =
        m_metall_manager->find<knn_index_container>(metall::unique_instance)
            .first;
    m_index_k_list =
        m_metall_manager->find<size_container>(metall::unique_instance).first;
  }

  /// \brief Add points to the internal point store.
  /// \tparam id_iterator Iterator type for point IDs.
  /// \tparam point_iterator Iterator type for points.
  /// \param ids_begin Iterator to the beginning of point IDs.
  /// \param ids_end Iterator to the end of point IDs.
  /// \param points_begin Iterator to the beginning of points.
  /// \param points_end Iterator to the end of points.
  template <typename id_iterator, typename point_iterator>
  void add_points(id_iterator ids_begin, id_iterator ids_end,
                  point_iterator points_begin, point_iterator points_end) {
    assert(m_pstore);
    m_pstore->reserve(std::distance(ids_begin, ids_end));
    for (auto id = ids_begin; id != ids_end; ++id) {
      assert(points_begin != points_end);
      (*m_pstore)[*id] = *points_begin;
      ++points_begin;
    }
  }

  /// \brief Load points from files and add to the internal point store.
  /// \tparam paths_iterator Iterator type for file paths.
  /// \param paths_begin Iterator to the beginning of file paths.
  /// \param paths_end Iterator to the end of file paths.
  /// \param file_format File format. Supported formats are 'csv' (CSV),
  /// 'csv-id' (CSV with IDs in the first column), 'wsv' (whitespace-separated
  /// values), and 'wsv-id' (whitespace-separated values with IDs in the first
  /// column).
  template <typename paths_iterator>
  void load_points(paths_iterator paths_begin, paths_iterator paths_end,
                   const std::string_view file_format) {
    static_assert(
        std::is_same_v<
            typename std::iterator_traits<paths_iterator>::value_type,
            std::filesystem::path>,
        "paths_iterator must be an iterator of std::filesystem::path");
    std::vector<std::string> point_file_paths;
    for (auto path = paths_begin; path != paths_end; ++path) {
      point_file_paths.push_back(path->string());
    }
    saltatlas::read_points(point_file_paths, file_format, false,
                           priv_get_point_partitioner(), *m_pstore, m_comm);
  }

  /// \brief Load points from files and add to the internal point store.
  /// This function assumes that there is one point per line.
  /// \tparam paths_iterator Iterator type for file paths.
  /// \param paths_begin Iterator to the beginning of file paths.
  /// \param paths_end Iterator to the end of file paths.
  /// \param line_parser A function that parses a line and returns a pair of
  /// point ID and point data.
  template <typename paths_iterator>
  void load_points(
      paths_iterator paths_begin, paths_iterator paths_end,
      const std::function<std::pair<id_type, point_type>(const std::string&)>&
          line_parser) {
    std::vector<std::string> point_file_paths(paths_begin, paths_end);

    const auto parser_wrapper = [&line_parser](const std::string& line,
                                               id_type& id, point_type& point) {
      auto ret = line_parser(line);
      id       = ret.first;
      point    = ret.second;
      return true;
    };

    saltatlas::dndetail::read_points_with_id_helper(
        point_file_paths, parser_wrapper, *m_pstore,
        priv_get_point_partitioner(), m_comm, false);
  }

  /// \brief Build a KNNG.
  /// \param k Number of neighbors per point.
  /// \param rho Rho parameter in NN-Descent.
  /// \param delta Delta parameter in NN-Descent.
  std::size_t build(distance_function_type dfunc, const int k,
                    const double rho = 0.8, const double delta = 0.001) {
    typename nn_kernel_type::option option{.k                          = k,
                                           .r                          = rho,
                                           .delta                      = delta,
                                           .exchange_reverse_neighbors = true,
                                           .mini_batch_size = 1 << 26,
                                           .rnd_seed        = m_rnd_seed,
                                           .verbose         = m_verbose};

    nn_kernel_type kernel(option, *m_pstore, priv_get_point_partitioner(),
                          dfunc, m_comm);
    m_knn_index_list->emplace_back();
    kernel.construct(m_knn_index_list->back());
    m_index_k_list->push_back(k);

    return m_knn_index_list->size() - 1;
  }

  /// \brief Build a KNNG.
  /// \param k Number of neighbors per point.
  /// \param initial_index Initial index.
  /// \param rho Rho parameter in NN-Descent.
  /// \param delta Delta parameter in NN-Descent.
  std::size_t build(distance_function_type dfunc, const int k,
                    const knn_index_type& initial_index, const double rho = 0.8,
                    const double delta = 0.001) {
    typename nn_kernel_type::option option{.k                          = k,
                                           .r                          = rho,
                                           .delta                      = delta,
                                           .exchange_reverse_neighbors = true,
                                           .mini_batch_size = 1 << 26,
                                           .rnd_seed        = m_rnd_seed,
                                           .verbose         = m_verbose};

    nn_kernel_type kernel(option, *m_pstore, priv_get_point_partitioner(),
                          dfunc, m_comm);
    m_knn_index_list->emplace_back();
    kernel.construct(initial_index, m_knn_index_list->back());
    m_index_k_list->push_back(k);

    return m_knn_index_list->size() - 1;
  }

  /// \brief Build a KNNG.
  /// \param k Number of neighbors per point.
  /// \param initial_index Initial index.
  /// \param rho Rho parameter in NN-Descent.
  /// \param delta Delta parameter in NN-Descent.
  std::size_t build(
      distance_function_type dfunc, const int k,
      const std::unordered_map<id_type, std::vector<id_type>>& initial_index,
      const double rho = 0.8, const double delta = 0.001) {
    typename nn_kernel_type::option option{.k                          = k,
                                           .r                          = rho,
                                           .delta                      = delta,
                                           .exchange_reverse_neighbors = true,
                                           .mini_batch_size = 1 << 26,
                                           .rnd_seed        = m_rnd_seed,
                                           .verbose         = m_verbose};

    nn_kernel_type kernel(option, *m_pstore, priv_get_point_partitioner(),
                          dfunc, m_comm);
    m_knn_index_list->emplace_back();
    kernel.construct(initial_index, m_knn_index_list->back());
    m_index_k_list->push_back(k);

    return m_knn_index_list->size() - 1;
  }

  /// \brief Update the KNNG.
  void update(const std::size_t index_id, distance_function_type dfunc,
              const int k, const double rho = 0.8, const double delta = 0.001) {
    typename nn_kernel_type::option option{.k                          = k,
                                           .r                          = rho,
                                           .delta                      = delta,
                                           .exchange_reverse_neighbors = true,
                                           .mini_batch_size = 1 << 26,
                                           .rnd_seed        = m_rnd_seed,
                                           .verbose         = m_verbose};

    nn_kernel_type kernel(option, *m_pstore, priv_get_point_partitioner(),
                          dfunc, m_comm);
    // TODO: implement more efficient update
    kernel.construct(m_index_k_list->at(index_id),
                     m_index_k_list->at(index_id));
    m_index_k_list->at(index_id) = k;
  }

  /// \brief Apply optimizations to an already constructed KNNG aiming at
  /// improving the query quality and performance.
  /// \param make_index_undirected If true, make the index undirected.
  /// \param make_index_undirected If true, make the graph undirected.
  /// \param pruning_degree_multiplier
  /// Each point keeps up to k * pruning_degree_multiplier nearest neighbors,
  /// where k is the number of neighbors each point in the index has.
  /// if this value is less than 0, there is no pruning.
  void optimize(const std::size_t      index_id,
                distance_function_type distance_function,
                const bool             make_index_undirected     = true,
                const double           pruning_degree_multiplier = 1.5) {
    assert(index_id < m_knn_index_list->size());
    const typename nn_index_optimizer_type::option opt{
        .index_k                   = m_index_k_list->at(index_id),
        .undirected                = make_index_undirected,
        .pruning_degree_multiplier = pruning_degree_multiplier,
        .remove_long_paths         = false,
        .verbose                   = m_verbose};
    nn_index_optimizer_type optimizer{opt,
                                      *m_pstore,
                                      priv_get_point_partitioner(),
                                      distance_function,
                                      m_knn_index_list->at(index_id),
                                      m_comm};
    optimizer.run();
  }

  /// \brief Query nearest neighbors of given points.
  /// This function assumes that the query points are already distributed.
  /// Query results are returned to the MPI rank that submitted the queries.
  /// \tparam query_iterator Iterator type for query points.
  /// \param queries_begin Iterator to the beginning of query points.
  /// \param queries_end Iterator to the end of query points.
  /// \param k The number of nearest neighbors to search for each point.
  /// \param epsilon The epsilon parameter in the search.
  /// \return Computed k nearest neighbors of the given points.
  /// Returned as an adjacency list (vector of vectors).
  /// Specifically, k nearest neighbor data of the i-th query is stored in the
  /// i-th inner vector. Each inner vector contains pairs of a neighbor ID and a
  /// distance to the neighbor from the query point.
  template <typename query_iterator>
  neighbor_store_type query(const std::size_t      index_id,
                            distance_function_type distance_function,
                            query_iterator         queries_begin,
                            query_iterator queries_end, const int k,
                            const double epsilon = 0.1) {
    typename query_kernel_type::option option{.k          = k,
                                              .epsilon    = epsilon,
                                              .mu         = 0,
                                              .batch_size = 1 << 26,
                                              .rnd_seed   = m_rnd_seed,
                                              .verbose    = m_verbose};

    query_kernel_type kernel(option, *m_pstore, priv_get_point_partitioner(),
                             distance_function, m_knn_index_list->at(index_id),
                             m_comm);

    query_store_type    queries(queries_begin, queries_end);
    neighbor_store_type query_result;
    kernel.query_batch(queries, query_result);

    return query_result;
  }

  template <typename index_id_iterator, typename query_iterator>
  neighbor_store_type query(index_id_iterator      index_ids_begin,
                            index_id_iterator      index_ids_end,
                            distance_function_type distance_function,
                            query_iterator         queries_begin,
                            query_iterator queries_end, const int k,
                            const double epsilon = 0.1) {
    typename query_kernel_type::option option{.k          = k,
                                              .epsilon    = epsilon,
                                              .mu         = 0,
                                              .batch_size = 1 << 26,
                                              .rnd_seed   = m_rnd_seed,
                                              .verbose    = m_verbose};

    auto tmp_index = m_knn_index_list->at(*index_ids_begin);
    for (auto index_id = index_ids_begin + 1; index_id != index_ids_end;
         ++index_id) {
      tmp_index.merge(m_knn_index_list->at(*index_id));
    }

    query_kernel_type kernel(option, *m_pstore, priv_get_point_partitioner(),
                             distance_function, tmp_index,
                             m_comm);

    query_store_type    queries(queries_begin, queries_end);
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
  void dump_graph(const std::size_t index_id, const std::filesystem::path& path,
                  const bool dump_distance = false) const {
    std::stringstream file_name;
    file_name << path.string() << "-" << m_comm.rank();
    const auto ret =
        m_knn_index_list->at(index_id).dump(file_name.str(), dump_distance);
  }

  /// \brief Check if the local point store contains a point with the given ID.
  /// \param id Point ID.
  bool contains_local(const id_type id) const { return m_pstore->contains(id); }

  /// \brief Get a point with the given ID from the local point store.
  const point_type& get_local_point(const id_type id) const {
    return m_pstore->at(id);
  }

  /// \brife Returns an iterator that points to the beginning of the locally
  /// stored points.
  auto local_points_begin() const { return m_pstore->begin(); }

  /// \brief Returns an iterator that points to the end of the locally stored
  /// points.
  auto local_points_end() const { return m_pstore->end(); }

  /// \brief API for using 'for_each' with local points.
  iterator_proxy_type local_points() const {
    return iterator_proxy_type(local_points_begin(), local_points_end());
  }

  /// \brief Erase a kNN index
  void erase(const id_type id) {
    m_knn_index_list->erase(id);
    m_index_k_list->erase(id);
  }

 private:
  /// \brief Return a point partitioner instance.
  /// \return A point partitioner instance.
  point_partitioner priv_get_point_partitioner() const {
    const int size = m_comm.size();
    return [size](const id_type& id) {
      return dndetail::murmurhash::hash<5981>{}(id) % size;
    };
  };

  ygm::comm&                       m_comm;
  uint64_t                         m_rnd_seed;
  bool                             m_verbose;
  std::unique_ptr<metall::manager> m_metall_manager{nullptr};
  point_store_type*                m_pstore{nullptr};
  knn_index_container*             m_knn_index_list{nullptr};
  size_container*                  m_index_k_list{nullptr};
};

}  // namespace saltatlas