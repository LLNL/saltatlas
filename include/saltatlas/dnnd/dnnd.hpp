// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef SALTATLAS_DNND_ADV_INCLUDED_HPP
#error \
    "saltatlas/dnnd/dnnd_adv.hpp is already included. Please include either saltatlas/dnnd/dnnd.hpp or saltatlas/dnnd/dnnd_adv.hpp, but not both."
#endif

#ifndef SALTATLAS_DNND_INCLUDED_HPP
#define SALTATLAS_DNND_INCLUDED_HPP
#endif  // SALTATLAS_DNND_INCLUDED_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <ygm/comm.hpp>
#include <ygm/container/detail/base_concepts.hpp>
#include <ygm/detail/collective.hpp>

#include "saltatlas/common/data_reader.hpp"
#include "saltatlas/common/detail/utilities/iterator_proxy.hpp"
#include "saltatlas/common/point_store.hpp"
#include "saltatlas/dnnd/detail/dnnd_kernel.hpp"
#include "saltatlas/dnnd/detail/nn_index.hpp"
#include "saltatlas/dnnd/detail/nn_index_optimizer.hpp"
#include "saltatlas/dnnd/detail/query_kernel.hpp"
#include "saltatlas/dnnd/distance.hpp"
#include "saltatlas/dnnd/feature_vector.hpp"
#include "saltatlas/dnnd/utility.hpp"

namespace saltatlas {

/// \brief Distributed NNDescent simple version.
/// \tparam Id Point ID type.
/// \tparam Point Point type.
/// \tparam Distance Distance type.
template <typename Id       = uint64_t,
          typename Point    = saltatlas::feature_vector<double>,
          typename Distance = double, typename IdHash = std::hash<Id>>
class dnnd {
 private:
  using self_type = dnnd<Id, Point, Distance, IdHash>;

  constexpr static unsigned int k_pstore_hash_seed            = 0xA1B2C3D4;
  constexpr static unsigned int k_point_partitioner_hash_seed = 0x1A2B3C4D;
  static_assert(k_pstore_hash_seed != k_point_partitioner_hash_seed,
                "k_pstore_hash_seed and k_point_partitioner_hash_seed must be "
                "different.");

 public:
  /// \brief Point ID type.
  using id_type = std::remove_cv_t<Id>;
  /// \brief Distance type.
  using distance_type = std::remove_cv_t<Distance>;
  /// \brief Point type.
  using point_type = Point;
  /// \brief Point ID hasher.
  using hasher = IdHash;

 private:
  /// \brief Internal ID type (contiguous integers starting from 0).
  using internal_id_type =
      std::conditional_t<std::is_integral_v<id_type>, id_type, uint64_t>;

  /// Use an ID table to map external IDs to internal IDs if id_type is not an
  /// integral type. If id_type is an integral type, we can use the ID directly
  /// as the internal
  static constexpr bool k_use_eid_table = !std::is_integral_v<id_type>;

  /// \brief Point store type.
  using point_store_type =
      point_store<internal_id_type, point_type, hash<k_pstore_hash_seed>,
                  std::equal_to<>, std::allocator<std::byte>>;
  using external_point_store_type =
      point_store<id_type, point_type, hash<k_pstore_hash_seed>,
                  std::equal_to<>, std::allocator<std::byte>>;

  /// \brief k-NN index type.
  using knn_index_type = dndetail::nn_index<internal_id_type, distance_type>;
  /// \brief k-NN index type with external ID type (i.e., id_type).
  using external_knn_index_type = dndetail::nn_index<id_type, distance_type>;

  using nn_kernel_type = dndetail::dnnd_kernel<point_store_type, distance_type>;

  /// \brief Point partitioner type.
  using internal_point_partitioner = typename nn_kernel_type::point_partitioner;

  using nn_index_optimizer_type =
      dndetail::nn_index_optimizer<point_store_type, knn_index_type>;

  using query_kernel_type =
      dndetail::dknn_batch_query_kernel<point_store_type, knn_index_type>;

  using query_store_type = typename query_kernel_type::query_store_type;

  using internal_neighbor_store_type =
      typename query_kernel_type::neighbor_store_type;

 public:
  /// \brief Neighbor type (contains a neighbor ID and the distance to the
  /// neighbor).
  using neighbor_type = detail::neighbor<id_type, distance_type>;

  using iterator_proxy_type =
      detail::iterator_proxy<typename point_store_type::const_iterator>;

  /// \brief Distance function type.
  /// Specifically, std::function<distance_type(const point_type &, const
  /// point_type &)>.
  using distance_function_type =
      distance::distance_function_type<point_type, distance_type>;

  /// \brief Query result store type. Specifically,
  /// std::vector<std::vector<neighbor_type>>.
  using neighbor_store_type = std::vector<std::vector<neighbor_type>>;

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
  /// All ranks must call this function even if some ranks add no points.
  /// \tparam id_iterator Iterator type for point IDs.
  /// \tparam point_iterator Iterator type for points.
  /// \param ids_begin Iterator to the beginning of point IDs.
  /// \param ids_end Iterator to the end of point IDs.
  /// \param points_begin Iterator to the beginning of points.
  /// \param points_end Iterator to the end of points.
  template <typename id_iterator, typename point_iterator>
  void add_points(id_iterator ids_begin, id_iterator ids_end,
                  point_iterator points_begin, point_iterator points_end) {
    static_assert(
        std::is_same_v<typename std::iterator_traits<id_iterator>::value_type,
                       id_type>,
        "id_iterator must be an iterator of id_type");
    static_assert(std::is_same_v<
                      typename std::iterator_traits<point_iterator>::value_type,
                      point_type>,
                  "point_iterator must be an iterator of point_type");

    priv_add_points(ids_begin, ids_end, points_begin, points_end);
    if (m_verbose) {
      m_comm.cout() << "Contains " << m_pstore.size()
                    << " points after adding points." << std::endl;
      m_comm.cf_barrier();
    }
  }

  /// \brief Add points to the internal point store.
  /// All ranks must call this function even if some ranks add no point.
  /// \tparam ygm_container_type Associative YGM container type for key-value
  /// store.
  /// \param container Associative YGM container.
  template <template <typename, typename> class ygm_container_type>
  void add_points(ygm_container_type<id_type, point_type>& container)
    requires ygm::container::detail::HasForAll<
                 ygm_container_type<id_type, point_type>> &&
             ygm::container::detail::DoubleItemTuple<
                 typename ygm_container_type<id_type, point_type>::for_all_args>
  {
    container.for_all([this](const id_type id, const point_type& point) {
      this->priv_add_point_async(id, point);
    });
    m_comm.barrier();
  }

  /// \brief Add points to the internal point store.
  /// All ranks must call this function even if some ranks add no point.
  /// \tparam ygm_container_type Associative YGM container type for key-value
  /// store (with array-type template signature).
  /// \param container Associative YGM container.
  template <template <typename, typename> class ygm_container_type>
  void add_points(ygm_container_type<point_type, id_type>& container)
    requires ygm::container::detail::HasForAll<
                 ygm_container_type<id_type, point_type>> &&
             ygm::container::detail::DoubleItemTuple<
                 typename ygm_container_type<id_type, point_type>::for_all_args>
  {
    container.for_all([this](const id_type id, const point_type& point) {
      this->priv_add_point_async(id, point);
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
  /// \note This function can be used with a point type that uses vector-like
  /// container, e.g., saltatlas::feature_vector.
  template <typename paths_iterator>
  void load_points(paths_iterator paths_begin, paths_iterator paths_end,
                   const std::string& file_format) {
    static_assert(
        std::is_same_v<
            typename std::iterator_traits<paths_iterator>::value_type,
            std::filesystem::path>,
        "paths_iterator must be an iterator of std::filesystem::path");

    std::vector<std::filesystem::path> point_file_paths(paths_begin, paths_end);
    if constexpr (k_use_eid_table) {
      external_point_store_type                pstore;
      std::vector<std::filesystem::path>       point_file_paths(paths_begin,
                                                                paths_end);
      const std::function<int(const id_type&)> partitioner =
          priv_get_point_partitioner_external();
      saltatlas::read_points(point_file_paths, file_format, m_verbose,
                             partitioner, pstore, m_comm);

      std::vector<id_type> eids;
      eids.reserve(pstore.size());
      for (const auto& [eid, _] : pstore) {
        eids.push_back(eid);
      }
      priv_gen_internal_id(eids.begin(), eids.end());

      for (const auto& [eid, point] : pstore) {
        priv_add_point_async(eid, point);
      }
      m_comm.barrier();
    } else {
      const std::function<int(const id_type&)> partitioner =
          priv_get_point_partitioner_internal();
      saltatlas::read_points(point_file_paths, file_format, m_verbose,
                             partitioner, m_pstore, m_comm);
    }
  }

  /// \brief Load points from files and add to the internal point store.
  /// This function assumes that there is one point per line.
  /// All ranks must call this function although some ranks load no points.
  /// \tparam paths_iterator Iterator type for file paths.
  /// \param paths_begin Iterator to the beginning of file paths.
  /// \param paths_end Iterator to the end of file paths.
  /// \param line_parser A function that parses a line and returns a pair of
  /// point ID and point data.
  /// \note This function can be used with a point type that uses vector-like
  /// container, e.g., saltatlas::feature_vector.
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

    if constexpr (k_use_eid_table) {
      external_point_store_type                pstore;
      std::vector<std::filesystem::path>       point_file_paths(paths_begin,
                                                                paths_end);
      const std::function<int(const id_type&)> partitioner =
          priv_get_point_partitioner_external();
      saltatlas::detail::read_points_with_id_helper(
          point_file_paths, parser_wrapper, pstore, partitioner, m_comm,
          m_verbose);

      std::vector<id_type> eids;
      eids.reserve(pstore.size());
      for (const auto& [eid, _] : pstore) {
        eids.push_back(eid);
      }
      priv_gen_internal_id(eids.begin(), eids.end());

      for (const auto& [eid, point] : pstore) {
        priv_add_point_async(eid, point);
      }
      m_comm.barrier();
    } else {
      const std::function<int(const id_type&)> partitioner =
          priv_get_point_partitioner_internal();
      saltatlas::detail::read_points_with_id_helper(
          point_file_paths, parser_wrapper, m_pstore, partitioner, m_comm,
          m_verbose);
    }
  }

  /// \brief Build a KNNG.
  /// All ranks must call this function.
  /// \param k Number of neighbors per point.
  /// \param rho Rho parameter in NN-Descent.
  /// \param delta Delta parameter in NN-Descent.
  /// \param batch_size Batch size parameter.
  /// \param time_limit_sec Timeout in seconds for the main neighbor check
  /// kernel. The elapsed time is checked after each neighbor check loop. If
  /// the time limit is exceeded, the construction stops. All ranks must use
  /// the same value. If 0 is given, there is no timeout.
  void build(const int k, const double rho = 0.5, const double delta = 0.001,
             const std::size_t batch_size     = 1 << 26,
             const double      time_limit_sec = 0) {
    typename nn_kernel_type::option option{.k              = k,
                                           .r              = rho,
                                           .delta          = delta,
                                           .time_limit_sec = time_limit_sec,
                                           .exchange_reverse_neighbors = true,
                                           .mini_batch_size = batch_size,
                                           .rnd_seed        = m_rnd_seed,
                                           .verbose         = m_verbose};

    nn_kernel_type kernel(option, m_pstore,
                          priv_get_point_partitioner_internal(),
                          m_distance_func, m_comm);
    kernel.construct(m_knn_index);
    m_index_k = k;
    if constexpr (k_use_eid_table) {
      priv_gen_external_knng();
    }
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
    nn_index_optimizer_type optimizer{opt,
                                      m_pstore,
                                      priv_get_point_partitioner_internal(),
                                      m_distance_func,
                                      m_knn_index,
                                      m_comm};
    optimizer.run();
    if constexpr (k_use_eid_table) {
      priv_gen_external_knng();
    }
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
  /// i-th inner vector. Each inner vector contains pairs of a neighbor ID and
  /// a distance to the neighbor from the query point.
  template <typename query_iterator>
  neighbor_store_type query(query_iterator queries_begin,
                            query_iterator queries_end, const int k,
                            const double epsilon = 0.1) {
    if constexpr (k_use_eid_table) {
      return priv_run_query(queries_begin, queries_end, k, epsilon).first;
    } else {
      return priv_run_query(queries_begin, queries_end, k, epsilon).second;
    }
  }

  /// \brief Run queries as the same as query() and return the query results
  /// with the feature vectors of the neighbors.
  template <typename query_iterator>
  std::pair<neighbor_store_type, std::vector<std::vector<point_type>>>
  query_with_features(query_iterator queries_begin, query_iterator queries_end,
                      const int k, const double epsilon = 0.1) {
    const auto query_results =
        priv_run_query(queries_begin, queries_end, k, epsilon);
    const auto& external_query_result = query_results.first;
    const auto& internal_query_result = query_results.second;

    const auto neighbor_features =
        priv_gather_neighbor_features(internal_query_result);

    if constexpr (k_use_eid_table) {
      return std::make_pair(std::move(external_query_result),
                            std::move(neighbor_features));
    } else {
      return std::make_pair(std::move(internal_query_result),
                            std::move(neighbor_features));
    }
  }

  /// \brief Dump the k-NN index to distributed files.
  /// \param out_file_prefix File path prefix.
  /// \param dump_distance If true, also dump distances
  /// \details For each neighbor list, the following lines are dumped:
  /// ```
  /// source_id neighbor_id_1 neighbor_id_2 ...
  /// 0.0 distance_1 distance_2 ...
  /// ```
  /// Each item is separated by a tab. The first line is the source id
  /// followed by neighbor ids. The second line is the distances to each
  /// neighbor. The first distance value is a dummy value (0.0), which is just
  /// a placeholder so that a neighbor id and the corresponding distance value
  /// is stored in the same column.
  /// \Note This function does not dump external point IDs.
  void dump_index(const std::filesystem::path& path,
                  const bool                   dump_distance = false) const {
    std::stringstream file_name;
    file_name << path.string() << "-" << m_comm.rank();
    if (k_use_eid_table) {
      m_external_knn_index.dump(file_name.str(), dump_distance);
    } else {
      m_knn_index.dump(file_name.str(), dump_distance);
    }
    m_comm.cf_barrier();
  }

  [[deprecated("Use dump_index() instead.")]]
  void dump_graph(const std::filesystem::path& path,
                  const bool                   dump_distance = false) const {
    dump_index(path, dump_distance);
  }

  /// \brief Check if the local point store contains a point with the given
  /// ID.
  /// \param id Point ID.
  bool contains_local(const id_type id) const {
    if constexpr (k_use_eid_table) {
      return m_local_e2i_id_table.contains(id);
    } else {
      return m_pstore.contains(id);
    }
  }

  /// \brief Get the owner rank of a point with the given ID.
  /// \param id Point ID.
  /// \return The rank that owns the point.
  int get_owner(const id_type id) const {
    static_assert(!k_use_eid_table,
                  "get_owner() is not available when external ID and internal "
                  "ID are different.");
    return priv_get_point_partitioner_internal()(id);
  }

  /// \brief Get a point of the given ID from the local point store.
  const point_type& get_local_point(const id_type id) const {
    const auto iid = priv_find_local_internal_id(id);
    return m_pstore.at(iid);
  }

  /// \brief Get point data of the given IDs.
  /// This function invokes YGM barrier. All ranks must call this function.
  template <typename id_iterator>
  std::unordered_map<id_type, point_type> get_points(
      id_iterator ids_begin, id_iterator ids_end) const {
    static_assert(
        std::is_same_v<typename std::iterator_traits<id_iterator>::value_type,
                       id_type>,
        "id_iterator must be an iterator of id_type");

    return priv_get_remote_points(ids_begin, ids_end);
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
  std::size_t num_points() const { return ygm::sum(m_pstore.size(), m_comm); }

  /// \brief Get the number of neighbors of the given point.
  /// If the point is not stored locally, the function returns 0.
  /// \param id Point ID.
  /// \return The number of neighbors of the point.
  std::size_t num_local_neighbors(const id_type id) const {
    if (contains_local(id)) {
      const auto iid = (k_use_eid_table) ? priv_find_local_internal_id(id) : id;
      return m_knn_index.num_neighbors(iid);
    }
    return 0;
  }

  /// \brief Get the neighbors of the given local point.
  /// If the point is not stored locally, the function throws an exception.
  /// \param id Point ID.
  /// \return The neighbors of the point. A vector of neighbor.
  std::vector<neighbor_type> get_local_neighbors(const id_type id) const {
    std::vector<neighbor_type> neighbors;
    if constexpr (k_use_eid_table) {
      for (auto itr = m_external_knn_index.neighbors_begin(id),
                end = m_external_knn_index.neighbors_end(id);
           itr != end; ++itr) {
        neighbors.push_back(*itr);
      }
    } else {
      const auto iid = priv_find_local_internal_id(id);
      for (auto itr = m_knn_index.neighbors_begin(iid),
                end = m_knn_index.neighbors_end(iid);
           itr != end; ++itr) {
        neighbors.push_back(*itr);
      }
    }

    return neighbors;
  }

  /// \brief Get the neighbors of the given point.
  /// This function invokes YGM barrier. All ranks must call this function.
  template <typename id_iterator>
  std::unordered_map<id_type, std::vector<neighbor_type>> get_neighbors(
      id_iterator ids_begin, id_iterator ids_en) const {
    static_assert(
        std::is_same_v<typename std::iterator_traits<id_iterator>::value_type,
                       id_type>,
        "id_iterator must be an iterator of id_type");

    static std::unordered_map<id_type, std::vector<neighbor_type>>
        neighbors_table;
    neighbors_table = decltype(neighbors_table){};
    neighbors_table.reserve(std::distance(ids_begin, ids_en));

    m_comm.cf_barrier();

    for (auto it = ids_begin; it != ids_en; ++it) {
      const auto id = *it;
      if constexpr (k_use_eid_table) {
        m_comm.async(
            priv_get_point_partitioner_external()(id),
            [](auto comm, auto pthis, const id_type eid,
               const int source_rank) {
              const auto iid = pthis->priv_find_local_internal_id(eid);
              comm->async(
                  pthis->priv_get_point_partitioner_internal()(iid),
                  [](auto comm, auto pthis, const auto& eid, const auto& iid,
                     const auto& source_rank) {
                    assert(pthis->m_pstore.contains(iid));
                    const auto neighbors = pthis->get_local_neighbors(eid);
                    comm->async(
                        source_rank,
                        [](auto, const auto& eid, const auto& neighbors) {
                          neighbors_table.emplace(eid, neighbors);
                        },
                        pthis->m_this, eid, neighbors);
                  },
                  pthis->m_this, eid, iid, source_rank);
            },
            m_this, id, m_comm.rank());
      } else {
        m_comm.async(
            priv_get_point_partitioner_internal()(id),
            [](auto comm, auto pthis, const id_type id, const int source_rank) {
              assert(pthis->m_pstore.contains(id));
              const auto neighbors = pthis->get_local_neighbors(id);
              comm->async(
                  source_rank,
                  [](auto, const auto& id, const auto& neighbors) {
                    neighbors_table.emplace(id, neighbors);
                  },
                  pthis->m_this, id, neighbors);
            },
            m_this, id, m_comm.rank());
      }
    }
    m_comm.barrier();

    return neighbors_table;
  }

  /// \brief Get the neighbors of the given point with features of the
  /// neighbors.
  template <typename id_iterator>
  std::unordered_map<
      id_type, std::pair<std::vector<neighbor_type>, std::vector<point_type>>>
  get_neighbors_with_features(id_iterator ids_begin,
                              id_iterator ids_end) const {
    // Get neighbors
    const auto neighbors_table = get_neighbors(ids_begin, ids_end);

    // Get neighbor's features
    std::set<id_type> neighbor_ids;
    for (auto& [id, neighbors] : neighbors_table) {
      for (const auto& neighbor : neighbors) {
        neighbor_ids.insert(neighbor.id);
      }
    }
    const auto neighbor_features_table =
        get_points(neighbor_ids.begin(), neighbor_ids.end());

    // Construct the result table
    std::unordered_map<
        id_type, std::pair<std::vector<neighbor_type>, std::vector<point_type>>>
        result;
    for (auto& [id, neighbors] : neighbors_table) {
      result[id];

      std::vector<point_type> neighbor_features;
      for (const auto& neighbor : neighbors) {
        neighbor_features.push_back(neighbor_features_table.at(neighbor.id));
      }

      result[id] =
          std::make_pair(std::move(neighbors), std::move(neighbor_features));
    }
    m_comm.cf_barrier();

    return result;
  }

 private:
  /// \brief Return a point partitioner instance.
  /// \return A point partitioner instance.
  auto priv_get_point_partitioner_external() const {
    const int size = m_comm.size();
    return [size](const id_type& id) {
      return hash<k_point_partitioner_hash_seed>{}(hasher{}(id)) % size;
    };
  };

  /// \brief Return a point partitioner instance.
  /// \return A point partitioner instance.
  internal_point_partitioner priv_get_point_partitioner_internal() const {
    const int size = m_comm.size();
    return [size](const internal_id_type& id) {
      return hash<k_point_partitioner_hash_seed>{}(id) % size;
    };
  };

  /// \brief Add points to the internal point store.
  template <typename id_iterator, typename point_iterator>
  void priv_add_points(id_iterator ids_begin, id_iterator ids_end,
                       point_iterator points_begin, point_iterator points_end) {
    if constexpr (k_use_eid_table) {
      priv_gen_internal_id(ids_begin, ids_end);
    }

    for (; ids_begin != ids_end; ++ids_begin, ++points_begin) {
      const auto& eid   = *ids_begin;
      const auto& point = *points_begin;
      priv_add_point_async(eid, point);
    }
    m_comm.barrier();
  }

  /// \brief Add a single point with external ID into pstore.
  void priv_add_point_async(const id_type& eid, const point_type& point) {
    const auto owner = priv_get_point_partitioner_external()(eid);

    if constexpr (k_use_eid_table) {
      // Get the internal ID corresponding to the given external ID.
      m_comm.async(
          owner,
          [](auto comm, auto pthis, const id_type& eid,
             const point_type& point) {
            const internal_id_type itn_id =
                pthis->priv_find_local_internal_id(eid);
            // Add the point with the internal ID into the point store.
            pthis->priv_add_point_with_internal_id_async(itn_id, eid, point);
          },
          m_this, eid, point);
    } else {
      m_comm.async(
          owner,
          [](auto comm, auto pthis, const id_type& id,
             const point_type& point) {
            pthis->priv_add_point_locally(id, point);
          },
          m_this, eid, point);
    }
  }

  /// \brief Add a single point with internal ID into pstore.
  void priv_add_point_with_internal_id_async(const internal_id_type& itn_id,
                                             const id_type&          eid,
                                             const point_type&       point) {
    static_assert(k_use_eid_table,
                  "priv_add_point_with_internal_id_async() is only available "
                  "when external "
                  "ID and internal ID are different.");

    auto receiver = [](auto, auto this_ptr, const internal_id_type id,
                       const auto& sent_point) {
      this_ptr->priv_add_point_locally(id, sent_point);
    };

    const auto owner = priv_get_point_partitioner_internal()(itn_id);
    m_comm.async(owner, receiver, m_this, itn_id, point);
  }

  void priv_add_point_locally(const internal_id_type& iid,
                              const point_type&       point) {
    if (m_pstore.contains(iid)) {
      std::cerr << "Duplicate internal ID " << iid << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    m_pstore[iid] = point;
  }

  // inline internal_id_type priv_find_or_make_local_internal_id(
  //     const id_type& eid) {
  //   if constexpr (!k_use_eid_table) {
  //     return eid;
  //   } else {
  //     if (m_e2i_id_table.find(eid) == m_e2i_id_table.end()) {
  //       const internal_id_type new_int_id = m_e2i_id_table.size();
  //       m_e2i_id_table[eid]               = new_int_id;
  //       return new_int_id;
  //     } else {
  //       return m_e2i_id_table.at(eid);
  //     }
  //   }
  //   assert(false);  // Should not reach here.
  //   return internal_id_type{};
  // }

  inline internal_id_type priv_find_local_internal_id(const id_type& id) const {
    if constexpr (!k_use_eid_table) {
      return id;
    } else {
      assert(m_e2i_id_table.contains(id));
      return m_e2i_id_table.at(id);
    }
  }

  inline id_type priv_get_local_external_id(const internal_id_type id) const {
    if constexpr (!k_use_eid_table) {
      return id;
    } else {
      return m_i2e_id_table.at(id);
    }
  }

  template <typename id_iterator>
  boost::unordered_flat_map<internal_id_type, id_type>
  priv_get_external_ids_async(id_iterator internal_ids_begin,
                              id_iterator internal_ids_end) const {
    static_assert(
        std::is_same_v<typename std::iterator_traits<id_iterator>::value_type,
                       internal_id_type>,
        "id_iterator must be an iterator of internal_id_type");

    static boost::unordered_flat_map<internal_id_type, id_type> return_id_table;
    return_id_table = decltype(return_id_table){};
    m_comm.cf_barrier();

    for (; internal_ids_begin != internal_ids_end; ++internal_ids_begin) {
      const auto internal_id = *internal_ids_begin;
      const auto owner = priv_get_point_partitioner_internal()(internal_id);
      m_comm.async(
          owner,
          [](auto comm, auto pthis, const internal_id_type internal_id,
             const int source_rank) {
            const auto external_id = pthis->m_i2e_id_table.at(internal_id);
            comm->async(
                source_rank,
                [](auto, const auto& internal_id, const auto& external_id) {
                  return_id_table[internal_id] = external_id;
                },
                internal_id, external_id);
          },
          m_this, internal_id, m_comm.rank());
    }
    m_comm.barrier();

    return return_id_table;
  }

  /// Generate consecutive internal IDs for the given external IDs and store
  /// the mapping in m_e2i_id_table and m_i2e_id_table.
  template <typename id_iterator>
  void priv_gen_internal_id(id_iterator eids_begin, id_iterator eids_end) {
    static_assert(
        k_use_eid_table,
        "priv_gen_internal_id() is only available when external ID and "
        "internal ID are different.");

    static_assert(
        std::is_same_v<typename std::iterator_traits<id_iterator>::value_type,
                       id_type>,
        "id_iterator must be an iterator of id_type");

    if constexpr (!k_use_eid_table) {
      return;
    }

    for (auto eit = eids_begin; eit != eids_end; ++eit) {
      const auto eid   = *eit;
      const auto owner = priv_get_point_partitioner_external()(eid);
      m_comm.async(
          owner,
          [](auto comm, auto pthis, const id_type eid) {
            // Assign a dummy internal ID for now
            pthis->m_e2i_id_table[eid] =
                std::numeric_limits<internal_id_type>::max();
          },
          m_this, eid);
    }
    m_comm.barrier();

    const size_t internal_id_offset =
        ygm::prefix_sum(m_e2i_id_table.size(), m_comm);

    // Assign internal IDs based on the computed offset
    size_t cnt = 0;
    for (auto& [eid, int_id] : m_e2i_id_table) {
      int_id = cnt + internal_id_offset;
      ++cnt;
    }

    // Tell the owner of each internal ID the corresponding external ID
    for (auto& [eid, int_id] : m_e2i_id_table) {
      const auto owner = priv_get_point_partitioner_internal()(int_id);
      m_comm.async(
          owner,
          [](auto comm, auto pthis, const internal_id_type int_id,
             const id_type eid) {
            pthis->m_i2e_id_table[int_id]    = eid;
            pthis->m_local_e2i_id_table[eid] = int_id;
          },
          m_this, int_id, eid);
    }
    m_comm.barrier();
  }

  void priv_gen_external_knng() {
    static_assert(
        k_use_eid_table,
        "priv_gen_external_knng() is only available when external ID and "
        "internal ID are different.");

    m_external_knn_index.reset();
    boost::unordered_flat_set<internal_id_type> internal_ids;
    for (auto& [internal_id, neighbors] : m_knn_index) {
      const auto external_id = priv_get_local_external_id(internal_id);
      for (const auto& neighbor : neighbors) {
        internal_ids.insert(neighbor.id);
      }
    }

    const auto i2e_id_map =
        priv_get_external_ids_async(internal_ids.begin(), internal_ids.end());
    m_comm.barrier();

    for (auto& [internal_id, neighbors] : m_knn_index) {
      const auto src_eid = priv_get_local_external_id(internal_id);
      for (const auto& neighbor : neighbors) {
        const auto n_eid = i2e_id_map.at(neighbor.id);
        m_external_knn_index.insert(src_eid,
                                    neighbor_type(n_eid, neighbor.distance));
      }
    }
  }

  template <typename query_iterator>
  std::pair<neighbor_store_type, internal_neighbor_store_type> priv_run_query(
      query_iterator queries_begin, query_iterator queries_end, const int k,
      const double epsilon = 0.1) {
    typename query_kernel_type::option option{.k          = k,
                                              .epsilon    = epsilon,
                                              .mu         = 0,
                                              .batch_size = 1 << 26,
                                              .rnd_seed   = m_rnd_seed,
                                              .verbose    = m_verbose};

    query_kernel_type kernel(option, m_pstore,
                             priv_get_point_partitioner_internal(),
                             m_distance_func, m_knn_index, m_comm);

    query_store_type             queries(queries_begin, queries_end);
    internal_neighbor_store_type internal_query_result;
    kernel.query_batch(queries, internal_query_result);

    if constexpr (k_use_eid_table) {
      return {priv_gen_external_query_result(internal_query_result),
              std::move(internal_query_result)};
    } else {
      return {neighbor_store_type{}, std::move(internal_query_result)};
    }
  }

  neighbor_store_type priv_gen_external_query_result(
      const internal_neighbor_store_type& internal_query_result) const {
    static_assert(
        k_use_eid_table,
        "priv_gen_external_query_result() is only available when external ID "
        "and internal ID are different.");

    boost::unordered_flat_set<internal_id_type> unique_neighbor_ids;
    for (const auto& neighbors : internal_query_result) {
      for (const auto& neighbor : neighbors) {
        unique_neighbor_ids.insert(neighbor.id);
      }
    }

    neighbor_store_type query_result;
    query_result.resize(internal_query_result.size());
    const auto i2e_id_map = priv_get_external_ids_async(
        unique_neighbor_ids.begin(), unique_neighbor_ids.end());
    for (std::size_t i = 0; i < internal_query_result.size(); ++i) {
      for (const auto& neighbor : internal_query_result[i]) {
        const auto eid = i2e_id_map.at(neighbor.id);
        query_result[i].emplace_back(eid, neighbor.distance);
      }
    }

    return query_result;
  }

  std::vector<std::vector<point_type>> priv_gather_neighbor_features(
      const internal_neighbor_store_type& query_result) const {
    std::vector<std::vector<point_type>> neighbor_features_store;
    neighbor_features_store.reserve(query_result.size());
    std::set<internal_id_type> neighbor_ids;
    for (const auto& neighbors : query_result) {
      for (const auto& neighbor : neighbors) {
        neighbor_ids.insert(neighbor.id);
      }
    }

    // Gather neighbor features
    static boost::unordered_flat_map<internal_id_type, point_type>
        neighbor_features;
    neighbor_features = decltype(neighbor_features){};
    m_comm.cf_barrier();
    for (const auto& niid : neighbor_ids) {
      const auto owner = priv_get_point_partitioner_internal()(niid);
      m_comm.async(
          owner,
          [](auto comm, auto pthis, const internal_id_type iid,
             const int source_rank) {
            assert(pthis->m_pstore.contains(iid));
            const auto& feature = pthis->m_pstore.at(iid);
            comm->async(
                source_rank,
                [](auto, const auto& iid, const auto& feature) {
                  neighbor_features[iid] = feature;
                },
                iid, feature);
          },
          m_this, niid, m_comm.rank());
    }
    m_comm.barrier();

    for (const auto& neighbors : query_result) {
      std::vector<point_type> neighbor_features_vec;
      neighbor_features_vec.reserve(neighbors.size());
      for (const auto& neighbor : neighbors) {
        neighbor_features_vec.push_back(neighbor_features.at(neighbor.id));
      }
      neighbor_features_store.push_back(std::move(neighbor_features_vec));
    }

    return neighbor_features_store;
  }

  template <typename id_iterator>
  std::unordered_map<id_type, point_type> priv_get_remote_points(
      id_iterator ids_begin, id_iterator ids_end) const {
    static std::unordered_map<id_type, point_type> return_points_store;
    return_points_store = decltype(return_points_store){};
    return_points_store.reserve(std::distance(ids_begin, ids_end));

    auto proc = [](auto comm, auto pthis, const id_type eid,
                   const int source_rank) {
      const auto iid = pthis->priv_find_local_internal_id(eid);
      comm->async(  // Move to the owner of the internal ID
          pthis->priv_get_point_partitioner_internal()(iid),
          [](auto comm, auto pthis, const auto& iid, const auto& eid,
             const auto& source_rank) {
            const auto& point = pthis->m_pstore.at(iid);
            comm->async(  // Move back to the source rank
                source_rank,
                [](auto, const auto& eid, const auto& point) {
                  return_points_store[eid] = point;
                },
                eid, point);
          },
          pthis->m_this, iid, eid, source_rank);
    };
    m_comm.cf_barrier();

    for (auto it = ids_begin; it != ids_end; ++it) {
      const auto id = *it;
      if constexpr (k_use_eid_table) {
        m_comm.async(priv_get_point_partitioner_external()(id), proc, m_this,
                     id, m_comm.rank());
      } else {
        m_comm.async(priv_get_point_partitioner_internal()(id), proc, m_this,
                     id, m_comm.rank());
      }
    }
    m_comm.barrier();

    return return_points_store;
  }

  distance_function_type  m_distance_func;
  ygm::comm&              m_comm;
  uint64_t                m_rnd_seed;
  point_store_type        m_pstore;
  knn_index_type          m_knn_index{};
  std::size_t             m_index_k{0};
  bool                    m_verbose;
  ygm::ygm_ptr<self_type> m_this{this};

  // Use the ID mapping tables only when id_type is not an integral type.
  // Note: Owing an external ID does not necessarily mean owning the
  // corresponding point data. The owner of a point data is determined by the
  // point partitioner, which is based on the internal ID.
  boost::unordered::unordered_flat_map<id_type, internal_id_type>
      m_e2i_id_table;
  // Contains external IDs of points stored locally.
  boost::unordered::unordered_flat_map<id_type, internal_id_type>
      m_local_e2i_id_table;
  boost::unordered::unordered_flat_map<internal_id_type, id_type>
                          m_i2e_id_table;
  external_knn_index_type m_external_knn_index{};
};

}  // namespace saltatlas
