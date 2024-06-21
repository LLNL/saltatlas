// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string_view>

#include <ygm/comm.hpp>

#include "saltatlas/dnnd/data_reader.hpp"
#include "saltatlas/dnnd/detail/base_dnnd.hpp"
#include "saltatlas/dnnd/detail/utilities/iterator_proxy.hpp"
#include "saltatlas/dnnd/feature_vector.hpp"

namespace saltatlas {

/// \brief Distributed NNDescent simple version.
/// \tparam Id Point ID type.
/// \tparam Point Point type.
/// \tparam Distance Distance type.
template <typename Id       = uint64_t,
          typename Point    = saltatlas::feature_vector<double>,
          typename Distance = double>
class dnnd {
 private:
  using base_type      = dndetail::base_dnnd<Id, Point, Distance>;
  using data_core_type = typename base_type::data_core_type;
  /// \brief Point store type.
  using point_store_type = typename base_type::point_store_type;
  /// \brief k-NN index type.
  using knn_index_type = typename base_type::knn_index_type;
  /// \brief Point partitioner type.
  using point_partitioner = typename base_type::point_partitioner;

 public:
  /// \brief Point ID type.
  using id_type = typename base_type::id_type;
  /// \brief Distance type.
  using distance_type = typename base_type::distance_type;
  /// \brief Point type.
  using point_type = typename base_type::point_type;
  /// \brief Neighbor type (contains a neighbor ID and the distance to the
  /// neighbor).
  using neighbor_type = typename base_type::neighbor_type;
  /// \brief Query result store type. Specifically,
  /// std::vector<std::vector<neighbor_type>>.
  using neighbor_store_type = typename base_type::neighbor_store_type;
  /// \brief Distance function type.
  /// Specifically, std::function<distance_type(const point_type &, const
  /// point_type &)>.
  using distance_function_type = typename base_type::distance_function_type;
  using iterator_proxy_type =
      dndetail::iterator_proxy<typename point_store_type::const_iterator>;

 private:
  using nn_kernel_type = dndetail::dnnd_kernel<point_store_type, distance_type>;
  using query_kernel_type =
      dndetail::dknn_batch_query_kernel<point_store_type, knn_index_type>;
  using nn_index_optimizer_type =
      dndetail::nn_index_optimizer<point_store_type, knn_index_type>;

 public:
  /// \brief Constructor.
  /// \param did Distance function id.
  /// \param comm YGM comm instance.
  /// \param rnd_seed Seed for random generators.
  /// \param verbose If true, enable the verbose mode.
  dnnd(const distance::id& did, ygm::comm& comm,
       const uint64_t rnd_seed = std::random_device{}(),
       const bool     verbose  = false)
      : m_base(verbose, comm), m_data_core(distance::id::custom, rnd_seed) {
    m_data_core.distance_id = did;
    m_base.set_data_core(m_data_core);
  }

  /// \brief Constructor.
  /// \param distance_func Distance function.
  /// \param comm YGM comm instance.
  /// \param rnd_seed Seed for random generators.
  /// \param verbose If true, enable the verbose mode.
  dnnd(const distance_function_type& distance_func, ygm::comm& comm,
       const uint64_t rnd_seed = std::random_device{}(),
       const bool     verbose  = false)
      : m_base(verbose, comm), m_data_core(distance::id::custom, rnd_seed) {
    m_base.set_data_core(m_data_core, distance_func);
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
    point_store_type& pstore = m_base.get_point_store();
    pstore.reserve(std::distance(ids_begin, ids_end));
    for (auto id = ids_begin; id != ids_end; ++id) {
      pstore[*id] = *points_begin;
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
                           m_base.get_point_partitioner(),
                           m_base.get_point_store(), m_base.get_comm());
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
        point_file_paths, parser_wrapper, m_base.get_point_store(),
        m_base.get_point_partitioner(), m_base.get_comm(), false);
  }

  /// \brief Build a KNNG.
  /// \param k Number of neighbors per point.
  /// \param rho Rho parameter in NN-Descent.
  /// \param delta Delta parameter in NN-Descent.
  void build(const int k, const double rho = 0.8, const double delta = 0.001) {
    m_base.construct_index(k, rho, delta, false, 1 << 28);
  }

  /// \brief Apply optimizations to an already constructed KNNG aiming at
  /// improving the query quality and performance.
  /// \param make_index_undirected If true, make the index undirected.
  /// \param make_index_undirected If true, make the graph undirected.
  /// \param pruning_degree_multiplier
  /// Each point keeps up to k * pruning_degree_multiplier nearest neighbors,
  /// where k is the number of neighbors each point in the index has.
  /// if this value is less than 0, there is no pruning.
  void optimize(const bool   make_index_undirected     = true,
                const double pruning_degree_multiplier = 1.5) {
    m_base.optimize_index(make_index_undirected, pruning_degree_multiplier,
                          false);
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
  neighbor_store_type query(query_iterator queries_begin,
                            query_iterator queries_end, const int k,
                            const double epsilon = 0.1) {
    std::vector<point_type> queries(queries_begin, queries_end);
    return m_base.query_batch(queries, k, epsilon, 0.0, 1 << 28);
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
  void dump_graph(const std::filesystem::path& path,
                  const bool                   dump_distance = false) const {
    m_base.dump_index(path.string(), dump_distance);
  }

  /// \brief Check if the local point store contains a point with the given ID.
  /// \param id Point ID.
  bool contains_local(const id_type id) const {
    return m_base.get_point_store().contains(id);
  }

  /// \brief Get a point with the given ID from the local point store.
  const point_type& get_local_point(const id_type id) const {
    return m_base.get_point_store().at(id);
  }

  /// \brife Returns an iterator that points to the beginning of the locally
  /// stored points.
  auto local_points_begin() const { return m_base.get_point_store().begin(); }

  /// \brief Returns an iterator that points to the end of the locally stored
  /// points.
  auto local_points_end() const { return m_base.get_point_store().end(); }

  // API for using 'for_each' with local points.
  iterator_proxy_type local_points() const {
    return iterator_proxy_type(local_points_begin(), local_points_end());
  }

 private:
  data_core_type m_data_core;
  base_type      m_base;
};

}  // namespace saltatlas