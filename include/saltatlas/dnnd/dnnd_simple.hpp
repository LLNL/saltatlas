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

#include <ygm/comm.hpp>

#include "saltatlas/dnnd/data_reader.hpp"
#include "saltatlas/dnnd/detail/distance.hpp"
#include "saltatlas/dnnd/detail/dnnd_kernel.hpp"
#include "saltatlas/dnnd/detail/nn_index.hpp"
#include "saltatlas/dnnd/detail/nn_index_optimizer.hpp"
#include "saltatlas/dnnd/detail/query_kernel.hpp"
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
 private:
  using self_type = dnnd<Id, Point, Distance>;

 public:
  /// \brief Point ID type.
  using id_type = Id;
  /// \brief Distance type.
  using distance_type = Distance;
  /// \brief Point type.
  using point_type = Point;

 private:
  /// \brief Point store type.
  using point_store_type =
      point_store<id_type, point_type, std::hash<id_type>, std::equal_to<>,
                  std::allocator<std::byte>>;
  /// \brief k-NN index type.
  using knn_index_type = dndetail::nn_index<id_type, distance_type>;

  using nn_kernel_type = dndetail::dnnd_kernel<point_store_type, distance_type>;

  /// \brief Point partitioner type.
  using point_partitioner = typename nn_kernel_type::point_partitioner;

  using nn_index_optimizer_type =
      dndetail::nn_index_optimizer<point_store_type, knn_index_type>;

  using query_kernel_type =
      dndetail::dknn_batch_query_kernel<point_store_type, knn_index_type>;

  using query_store_type = typename query_kernel_type::query_store_type;

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

 public:
  /// \brief Query result store type. Specifically,
  /// std::vector<std::vector<neighbor_type>>.
  using neighbor_store_type = typename query_kernel_type::neighbor_store_type;

  /// \brief Constructor.
  /// \param distance_func_id Distance function id.
  /// \param comm YGM comm instance.
  /// \param rnd_seed Seed for random generators.
  /// \param verbose If true, enable the verbose mode.
  dnnd(const distance::id& distance_func_id, ygm::comm& comm,
       const uint64_t rnd_seed = std::random_device{}(),
       const bool     verbose  = false)
      : m_distance_func(distance::distance_function<point_type, distance_type>(
            distance_func_id)),
        m_comm(comm),
        m_rnd_seed(rnd_seed),
        m_verbose(verbose) {
    m_comm.cf_barrier();
  }

  /// \brief Constructor.
  /// \param distance_func Distance function.
  /// \param comm YGM comm instance.
  /// \param rnd_seed Seed for random generators.
  /// \param verbose If true, enable the verbose mode.
  dnnd(const distance_function_type& distance_func, ygm::comm& comm,
       const uint64_t rnd_seed = std::random_device{}(),
       const bool     verbose  = false)
      : m_distance_func(distance_func),
        m_comm(comm),
        m_rnd_seed(rnd_seed),
        m_verbose(verbose) {
    m_comm.cf_barrier();
  }

  /// \brief Add points to the internal point store.
  /// All ranks must call this function although some ranks add no points.
  /// \tparam id_iterator Iterator type for point IDs.
  /// \tparam point_iterator Iterator type for points.
  /// \param ids_begin Iterator to the beginning of point IDs.
  /// \param ids_end Iterator to the end of point IDs.
  /// \param points_begin Iterator to the beginning of points.
  /// \param points_end Iterator to the end of points.
  template <typename id_iterator, typename point_iterator>
  void add_points(id_iterator ids_begin, id_iterator ids_end,
                  point_iterator points_begin, point_iterator points_end) {
    auto receiver = [](auto, auto this_ptr, const id_t id,
                       const auto& sent_point) {
      if (this_ptr->m_pstore.contains(id)) {
        std::cerr << "Duplicate ID " << id << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      this_ptr->m_pstore[id] = sent_point;
    };

    for (; ids_begin != ids_end; ++ids_begin, ++points_begin) {
      const auto dst = priv_get_point_partitioner()(*ids_begin);
      m_comm.async(dst, receiver, m_this, *ids_begin, *points_begin);
    }
    m_comm.barrier();
  }

  /// \brief Add points to the internal point store.
  /// All ranks must call this function although some ranks add no points.
  /// \tparam container_type Associative YGM container type for key-value store.
  /// \param container Associative YGM container.
  template <template <typename, typename> class container_type>
  void add_points(container_type<id_type, point_type>& container) {
    container.for_all([this](const id_type id, const point_type& point) {
      this->add_point(id, point);
    });
    m_comm.barrier();
  }

  /// \brief Load points from files and add to the internal point store.
  /// All ranks must call this function although some ranks load no points.
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
    std::vector<std::filesystem::path> point_file_paths(paths_begin, paths_end);
    saltatlas::read_points(point_file_paths, file_format, false,
                           priv_get_point_partitioner(), m_pstore, m_comm);
  }

  /// \brief Load points from files and add to the internal point store.
  /// This function assumes that there is one point per line.
  /// All ranks must call this function although some ranks load no points.
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
    std::vector<std::filesystem::path> point_file_paths(paths_begin, paths_end);

    const auto parser_wrapper = [&line_parser](const std::string& line,
                                               id_type& id, point_type& point) {
      auto ret = line_parser(line);
      id       = ret.first;
      point    = ret.second;
      return true;
    };

    saltatlas::dndetail::read_points_with_id_helper(
        point_file_paths, parser_wrapper, m_pstore,
        priv_get_point_partitioner(), m_comm, false);
  }

  /// \brief Build a KNNG.
  /// All ranks must call this function.
  /// \param k Number of neighbors per point.
  /// \param rho Rho parameter in NN-Descent.
  /// \param delta Delta parameter in NN-Descent.
  void build(const int k, const double rho = 0.8, const double delta = 0.001,
             const std::size_t batch_size = 1 << 26) {
    typename nn_kernel_type::option option{.k                          = k,
                                           .r                          = rho,
                                           .delta                      = delta,
                                           .exchange_reverse_neighbors = true,
                                           .mini_batch_size = batch_size,
                                           .rnd_seed        = m_rnd_seed,
                                           .verbose         = m_verbose};

    nn_kernel_type kernel(option, m_pstore, priv_get_point_partitioner(),
                          m_distance_func, m_comm);
    kernel.construct(m_knn_index);
    m_index_k = k;
  }

  /// \brief Apply optimizations to an already constructed KNNG aiming at
  /// improving the query quality and performance.
  /// All ranks must call this function.
  /// \param make_index_undirected If true, make the index undirected.
  /// \param make_index_undirected If true, make the graph undirected.
  /// \param pruning_degree_multiplier
  /// Each point keeps up to k * pruning_degree_multiplier nearest neighbors,
  /// where k is the number of neighbors each point in the index has.
  /// if this value is less than 0, there is no pruning.
  void optimize(const bool   make_index_undirected     = true,
                const double pruning_degree_multiplier = 1.5) {
    const typename nn_index_optimizer_type::option opt{
        .index_k                   = m_index_k,
        .undirected                = make_index_undirected,
        .pruning_degree_multiplier = pruning_degree_multiplier,
        .remove_long_paths         = false,
        .verbose                   = m_verbose};
    nn_index_optimizer_type optimizer{
        opt,         m_pstore, priv_get_point_partitioner(), m_distance_func,
        m_knn_index, m_comm};
    optimizer.run();
  }

  /// \brief Query nearest neighbors of given points.
  /// This function assumes that the query points are already distributed.
  /// Query results are returned to the MPI rank that submitted the queries.
  /// All ranks must call this function.
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
    typename query_kernel_type::option option{.k          = k,
                                              .epsilon    = epsilon,
                                              .mu         = 0,
                                              .batch_size = 1 << 26,
                                              .rnd_seed   = m_rnd_seed,
                                              .verbose    = m_verbose};

    query_kernel_type kernel(option, m_pstore, priv_get_point_partitioner(),
                             m_distance_func, m_knn_index, m_comm);

    query_store_type    queries(queries_begin, queries_end);
    neighbor_store_type query_result;
    kernel.query_batch(queries, query_result);

    return query_result;
  }

  template <typename query_iterator>
  std::pair<neighbor_store_type, std::vector<std::vector<point_type>>>
  query_with_features(query_iterator queries_begin, query_iterator queries_end,
                      const int k, const double epsilon = 0.1) {
    auto query_result = query(queries_begin, queries_end, k, epsilon);

    std::vector<std::vector<point_type>> neighbor_features_store;
    neighbor_features_store.reserve(query_result.size());
    for (const auto& neighbors : query_result) {
      std::vector<id_type> neighbor_ids;
      neighbor_ids.reserve(neighbors.size());
      for (const auto& neighbor : neighbors) {
        neighbor_ids.push_back(neighbor.id);
      }
      auto neighbor_features =
          get_points(neighbor_ids.begin(), neighbor_ids.end());

      std::vector<point_type> neighbor_features_vec;
      neighbor_features_vec.reserve(neighbor_ids.size());
      for (const auto& id : neighbor_ids) {
        neighbor_features_vec.push_back(neighbor_features.at(id));
      }
      neighbor_features_store.push_back(std::move(neighbor_features_vec));
    }

    return std::make_pair(query_result, neighbor_features_store);
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
    std::stringstream file_name;
    file_name << path.string() << "-" << m_comm.rank();
    m_knn_index.dump(file_name.str(), dump_distance);
  }

  /// \brief Check if the local point store contains a point with the given ID.
  /// \param id Point ID.
  bool contains_local(const id_type id) const { return m_pstore.contains(id); }

  /// \brief Get the owner rank of a point with the given ID.
  /// \param id Point ID.
  /// \return The rank that owns the point.
  int get_owner(const id_type id) const {
    return priv_get_point_partitioner()(id);
  }

  /// \brief Get a point with the given ID from the local point store.
  const point_type& get_local_point(const id_type id) const {
    return m_pstore.at(id);
  }

  template <typename id_iterator>
  std::unordered_map<id_type, point_type> get_local_points(
      id_iterator ids_begin, id_iterator ids_end) const {
    static_assert(
        std::is_same_v<typename std::iterator_traits<id_iterator>::value_type,
                       id_type>,
        "id_iterator must be an iterator of id_type");

    std::unordered_map<id_type, point_type> points;
    points.reserve(std::distance(ids_begin, ids_end));
    for (auto it = ids_begin; it != ids_end; ++it) {
      const auto id = *it;
      points.emplace(id, m_pstore.at(id));
    }
    return points;
  }

  template <typename id_iterator>
  std::unordered_map<id_type, point_type> get_points(
      id_iterator ids_begin, id_iterator ids_end) const {
    static_assert(
        std::is_same_v<typename std::iterator_traits<id_iterator>::value_type,
                       id_type>,
        "id_iterator must be an iterator of id_type");

    std::unordered_map<id_type, point_type> points;
    points.reserve(std::distance(ids_begin, ids_end));

    static auto& ref_points = points;
    ref_points              = points;

    auto proc = [](auto comm, auto pthis, const id_type id,
                   const int source_rank) {
      assert(pthis->contains_local(id));

      comm->async(
          source_rank,
          [](auto, const auto& id, const auto& point) {
            ref_points.emplace(id, point);
          },
          id, pthis->get_local_point(id));
    };

    for (auto it = ids_begin; it != ids_end; ++it) {
      const auto id = *it;
      m_comm.async(get_owner(id), proc, m_this, id, m_comm.rank());
    }

    m_comm.barrier();

    return points;
  }

  /// \brife Returns an iterator that points to the beginning of the locally
  /// stored points.
  auto local_points_begin() const { return m_pstore.begin(); }

  /// \brief Returns an iterator that points to the end of the locally stored
  /// points.
  auto local_points_end() const { return m_pstore.end(); }

  // API for using 'for_each' with local points.
  iterator_proxy_type local_points() const {
    return iterator_proxy_type(local_points_begin(), local_points_end());
  }

  /// \brief Get the number of locally stored points.
  std::size_t num_local_points() const { return m_pstore.size(); }

  /// \brief Get the number of points.
  /// This function performs an all-reduce operation, which is not cheap.
  std::size_t num_points() const {
    return m_comm.all_reduce_sum(m_pstore.size());
  }

 private:
  /// \brief Return a point partitioner instance.
  /// \return A point partitioner instance.
  point_partitioner priv_get_point_partitioner() const {
    const int size = m_comm.size();
    // TODO: hash id?
    return [size](const id_type& id) { return id % size; };
  };

  /// \brief Add a single point. Only to be used by add_points.
  void add_point(const id_type id, const point_type& point) {
    auto receiver = [](auto, auto this_ptr, const id_t id,
                       const auto& sent_point) {
      if (this_ptr->m_pstore.contains(id)) {
        std::cerr << "Duplicate ID " << id << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      this_ptr->m_pstore[id] = sent_point;
    };

    const auto dst = priv_get_point_partitioner()(id);
    m_comm.async(dst, receiver, m_this, id, point);
  }

  distance_function_type  m_distance_func;
  ygm::comm&              m_comm;
  uint64_t                m_rnd_seed;
  point_store_type        m_pstore;
  knn_index_type          m_knn_index{};
  std::size_t             m_index_k{0};
  bool                    m_verbose;
  ygm::ygm_ptr<self_type> m_this{this};
};

}  // namespace saltatlas