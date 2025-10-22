// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef SALTATLAS_DNND_INCLUDED_HPP
#error \
    "saltatlas/dnnd/dnnd.hpp is already included. Please include either saltatlas/dnnd/dnnd.hpp or saltatlas/dnnd/dnnd_adv.hpp, but not both."
#endif

#ifndef SALTATLAS_DNND_ADV_INCLUDED_HPP
#define SALTATLAS_DNND_ADV_INCLUDED_HPP
#endif  // SALTATLAS_DNND_ADV_INCLUDED_HPP

#include <algorithm>
#include <filesystem>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <string_view>

#include <boost/interprocess/containers/stable_vector.hpp>
#include <metall/container/vector.hpp>
#include <metall/utility/metall_mpi_adaptor.hpp>

#include <ygm/comm.hpp>
#include <ygm/detail/collective.hpp>

#include "saltatlas/common/data_reader.hpp"
#include "saltatlas/common/detail/utilities/hash.hpp"
#include "saltatlas/common/detail/utilities/iterator_proxy.hpp"
#include "saltatlas/common/point_store.hpp"
#include "saltatlas/dnnd/detail/dnnd_kernel.hpp"
#include "saltatlas/dnnd/detail/nn_index.hpp"
#include "saltatlas/dnnd/detail/nn_index_optimizer.hpp"
#include "saltatlas/dnnd/detail/query_kernel.hpp"
#include "saltatlas/dnnd/distance.hpp"
#include "saltatlas/dnnd/feature_vector.hpp"

namespace saltatlas {

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

/// \brief Distributed NNDescent simple version.
/// \tparam Id Point ID type.
/// \tparam Point Point type.
/// \tparam Distance Distance type.
template <typename Id       = uint64_t,
          typename Point    = saltatlas::pm_feature_vector<double>,
          typename Distance = double>
class dnnd_adv {
 private:
  using self_type = dnnd_adv<Id, Point, Distance>;
  using mmanager  = metall::utility::metall_mpi_adaptor::manager_type;

  template <typename T>
  using allocator_type = mmanager::fallback_allocator<T>;

  template <typename T>
  using scp_allocator_type = mmanager::scoped_fallback_allocator_type<T>;

 public:
  /// \brief Point ID type.
  using id_type = Id;
  /// \brief Distance type.
  using distance_type = Distance;
  /// \brief Point type.
  using point_type = Point;

  /// \brief k-NN index type.
  using knn_index_type =
      dndetail::nn_index<id_type, distance_type, allocator_type<std::byte>>;

 private:
  /// \brief Point store type.
  using point_store_type =
      point_store<id_type, point_type, std::hash<id_type>, std::equal_to<>,
                  allocator_type<std::byte>>;

  using nn_kernel_type = dndetail::dnnd_kernel<point_store_type, distance_type>;

  /// \brief Point partitioner type.
  using point_partitioner = typename nn_kernel_type::point_partitioner;

  using nn_index_optimizer_type =
      dndetail::nn_index_optimizer<point_store_type, knn_index_type>;

  using query_kernel_type =
      dndetail::dknn_batch_query_kernel<point_store_type, knn_index_type>;

  using query_store_type = typename query_kernel_type::query_store_type;

  using knn_index_container =
      boost::interprocess::stable_vector<knn_index_type,
                                         scp_allocator_type<knn_index_type>>;
  using size_container =
      metall::container::vector<std::size_t, scp_allocator_type<std::size_t>>;

 public:
  /// \brief Neighbor type (contains a neighbor ID and the distance to the
  /// neighbor).
  using neighbor_type = typename knn_index_type::neighbor_type;

  using iterator_proxy_type =
      detail::iterator_proxy<typename point_store_type::const_iterator>;

  /// \brief Distance function type.
  /// Specifically, std::function<distance_type(const point_type &, const
  /// point_type &)>.
  using distance_function_type =
      distance::distance_function_type<point_type, distance_type>;

  /// \brief Query result store type. Specifically,
  /// std::vector<std::vector<neighbor_type>>.
  using neighbor_store_type = typename query_kernel_type::neighbor_store_type;

  /// \brief Return the owner rank of the given point ID.
  static constexpr int get_owner(const id_type& id, const int mpi_size) {
    return hash<5981>{}(id) % mpi_size;
  }

  static bool copy(const std::filesystem::path& src_path,
                   const std::filesystem::path& dst_path, ygm::comm& comm,
                   const bool verbose = false) {
    const auto ret = metall::utility::metall_mpi_adaptor::copy(
        src_path, dst_path, comm.get_mpi_comm(), true);
    comm.barrier();
    if (!ret) {
      if (comm.rank0()) {
        std::cerr << "Failed to copy Metall datastore from " << src_path
                  << " to " << dst_path << std::endl;
      }
      return false;
    }
    if (verbose) {
      comm.cout0() << "Copied PM datastore from " << src_path << " to "
                   << dst_path << std::endl;
    }
    return true;
  }

  /// \brief Constructor. This constructor allocates data structures on DRAM,
  /// which are not persistent.
  /// \param comm YGM comm instance.
  /// \param rnd_seed Seed for random generators.
  /// \param verbose If true, enable the verbose mode.
  explicit dnnd_adv(ygm::comm&     comm,
                    const uint64_t rnd_seed = std::random_device{}(),
                    const bool     verbose  = false)
      : m_comm(comm), m_rnd_seed(rnd_seed), m_verbose(verbose) {
    m_pstore         = std::make_unique<point_store_type>();
    m_knn_index_list = std::make_unique<knn_index_container>();
    m_index_k_list   = std::make_unique<size_container>();
  }

  /// \brief Constructor. This constructor creates a persistent (Metall)
  /// datastore to store data structures (e.g., point store and knng index).
  /// \param datastore_path Filesystem path to the Metall datastore.
  /// \param comm YGM comm instance.
  /// \param rnd_seed Seed for random generators.
  /// \param verbose If true, enable the verbose mode.
  dnnd_adv(create_only_t, const std::filesystem::path& datastore_path,
           ygm::comm& comm, const uint64_t rnd_seed = std::random_device{}(),
           const bool verbose = false)
      : m_comm(comm), m_rnd_seed(rnd_seed), m_verbose(verbose) {
    m_metall = std::make_unique<metall::utility::metall_mpi_adaptor>(
        metall::create_only, datastore_path.string(), m_comm.get_mpi_comm(),
        true);
    auto& localm = m_metall->get_local_manager();
    m_pstore.reset(localm.construct<point_store_type>(metall::unique_instance)(
        localm.get_allocator<>()));
    m_knn_index_list.reset(localm.construct<knn_index_container>(
        metall::unique_instance)(localm.get_allocator<>()));
    m_index_k_list.reset(localm.construct<size_container>(
        metall::unique_instance)(localm.get_allocator<>()));
    m_comm.cf_barrier();
  }

  /// \brief Constructor. This constructor opens an existing persistent (Metall)
  /// datastore.
  dnnd_adv(open_only_t, const std::filesystem::path& datastore_path,
           ygm::comm& comm, const uint64_t rnd_seed = std::random_device{}(),
           const bool verbose = false)
      : m_comm(comm), m_rnd_seed(rnd_seed), m_verbose(verbose) {
    m_metall = std::make_unique<metall::utility::metall_mpi_adaptor>(
        metall::open_only, datastore_path.string(), m_comm.get_mpi_comm());
    auto& localm = m_metall->get_local_manager();
    m_pstore.reset(
        localm.find<point_store_type>(metall::unique_instance).first);
    assert(m_pstore);
    m_knn_index_list.reset(
        localm.find<knn_index_container>(metall::unique_instance).first);
    assert(m_knn_index_list);
    m_index_k_list.reset(
        localm.find<size_container>(metall::unique_instance).first);
    assert(m_index_k_list);
    m_comm.cf_barrier();
  }

  /// \brief Constructor. This constructor opens an existing persistent (Metall)
  /// datastore in read-only mode.
  dnnd_adv(open_read_only_t, const std::filesystem::path& datastore_path,
           ygm::comm& comm, const uint64_t rnd_seed = std::random_device{}(),
           const bool verbose = false)
      : m_comm(comm), m_rnd_seed(rnd_seed), m_verbose(verbose) {
    m_metall = std::make_unique<metall::utility::metall_mpi_adaptor>(
        metall::open_read_only, datastore_path.string(), m_comm.get_mpi_comm());
    auto& localm = m_metall->get_local_manager();
    m_pstore.reset(
        localm.find<point_store_type>(metall::unique_instance).first);
    assert(m_pstore);
    m_knn_index_list.reset(
        localm.find<knn_index_container>(metall::unique_instance).first);
    assert(m_knn_index_list);
    m_index_k_list.reset(
        localm.find<size_container>(metall::unique_instance).first);
    assert(m_index_k_list);
    m_comm.cf_barrier();
  }

  ~dnnd_adv() noexcept {
    if (m_metall) {
      // To keep the objects in the Metall datastore, do not destroy them.
      m_pstore.release();
      m_knn_index_list.release();
      m_index_k_list.release();
    }
  }

  /// \brief Add points to the internal point store.
  /// All ranks must call this function.
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
      if ((this_ptr->m_pstore)->contains(id)) {
        std::cerr << "Duplicate ID " << id << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      (*(this_ptr->m_pstore))[id] = sent_point;
    };

    for (; ids_begin != ids_end; ++ids_begin, ++points_begin) {
      const auto dst = priv_get_point_partitioner()(*ids_begin);
      m_comm.async(dst, receiver, m_this, *ids_begin, *points_begin);
    }
    m_comm.barrier();
  }

  /// \brief Load points from files and add to the internal point store.
  /// All ranks must call this function.
  /// \tparam paths_iterator Iterator type for file paths.
  /// \param paths_begin Iterator to the beginning of file paths.
  /// \param paths_end Iterator to the end of file paths.
  /// \param file_format File format. Supported formats are 'csv' (CSV),
  /// 'csv-id' (CSV with IDs in the first column), 'wsv' (whitespace-separated
  /// values), and 'wsv-id' (whitespace-separated values with IDs in the first
  /// column).
  template <typename paths_iterator>
  void load_points(paths_iterator paths_begin, paths_iterator paths_end,
                   const std::string& file_format) {
    static_assert(
        std::is_same_v<
            typename std::iterator_traits<paths_iterator>::value_type,
            std::filesystem::path>,
        "paths_iterator must be an iterator of std::filesystem::path");
    std::vector<std::filesystem::path> point_file_paths;
    for (auto path = paths_begin; path != paths_end; ++path) {
      point_file_paths.push_back(path->string());
    }
    saltatlas::read_points(point_file_paths, file_format, false,
                           priv_get_point_partitioner(), *m_pstore, m_comm);
  }

  /// \brief Load points from files and add to the internal point store.
  /// All ranks must call this function.
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
    std::vector<std::filesystem::path> point_file_paths(paths_begin, paths_end);

    const auto parser_wrapper = [&line_parser](const std::string& line,
                                               id_type& id, point_type& point) {
      auto ret = line_parser(line);
      id       = ret.first;
      point    = ret.second;
      return true;
    };

    saltatlas::detail::read_points_with_id_helper(
        point_file_paths, parser_wrapper, *m_pstore,
        priv_get_point_partitioner(), m_comm, false);
  }

  /// \brief Build a KNNG.
  /// All ranks must call this function.
  /// \param distance_func_id Distance function ID.
  /// \param k Number of neighbors per point.
  /// \param rho Rho parameter in NN-Descent.
  /// \param delta Delta parameter in NN-Descent.
  /// \param time_limit_sec Timeout in seconds for the main neighbor check
  /// kernel. The elapsed time is checked after each neighbor check loop. If the
  /// time limit is exceeded, the construction stops. All ranks must use the
  /// same value. If 0 is given, there is no timeout.
  std::size_t build(const distance::id& distance_func_id, const int k,
                    const double rho = 0.5, const double delta = 0.001,
                    const double time_limit_sec = 0) {
    return build(distance::distance_function<point_type, distance_type>(
                     distance_func_id),
                 k, rho, delta, time_limit_sec);
  }

  /// \brief Build a KNNG.
  /// All ranks must call this function.
  /// \param dfunc Distance function.
  /// \param k Number of neighbors per point.
  /// \param rho Rho parameter in NN-Descent.
  /// \param delta Delta parameter in NN-Descent.
  /// \param time_limit_sec Timeout in seconds for the main neighbor check
  /// kernel. The elapsed time is checked after each neighbor check loop. If the
  /// time limit is exceeded, the construction stops. All ranks must use the
  /// same value. If 0 is given, there is no timeout.
  std::size_t build(distance_function_type dfunc, const int k,
                    const double rho = 0.5, const double delta = 0.001,
                    const double time_limit_sec = 0) {
    typename nn_kernel_type::option option{.k              = k,
                                           .r              = rho,
                                           .delta          = delta,
                                           .time_limit_sec = time_limit_sec,
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
  /// All ranks must call this function.
  /// \param distance_func_id Distance function ID.
  /// \param k Number of neighbors per point.
  /// \param initial_index Initial index. The return value of get_index()
  /// can be used.
  /// \param rho Rho parameter in NN-Descent.
  /// \param delta Delta parameter in NN-Descent.
  /// \param recheck If true, redo the neighbor check for the initial index,
  /// i.e., mark the initial neighbors as 'new' neighbors.
  /// \param time_limit_sec Timeout in seconds for the main neighbor check
  /// kernel. The elapsed time is checked after each neighbor check loop. If the
  /// time limit is exceeded, the construction stops. All ranks must use the
  /// same value. If 0 is given, there is no timeout.
  std::size_t build(const distance::id& distance_func_id, const int k,
                    const knn_index_type& initial_index, const double rho = 0.5,
                    const double delta = 0.001, const bool recheck = false,
                    const double time_limit_sec = 0) {
    return build(distance::distance_function<point_type, distance_type>(
                     distance_func_id),
                 k, initial_index, rho, delta, recheck, time_limit_sec);
  }

  /// \brief Build a KNNG.
  /// All ranks must call this function.
  /// \param dfunc Distance function.
  /// \param k Number of neighbors per point.
  /// \param initial_index Initial index. The return value of get_index()
  /// can be used.
  /// \param rho Rho parameter in NN-Descent.
  /// \param delta Delta parameter in NN-Descent.
  /// \param recheck If true, redo the neighbor check for the initial index,
  /// i.e., mark the initial neighbors as 'new' neighbors.
  /// \param time_limit_sec Timeout in seconds for the main neighbor check
  /// kernel. The elapsed time is checked after each neighbor check loop. If the
  /// time limit is exceeded, the construction stops. All ranks must use the
  /// same value. If 0 is given, there is no timeout.
  std::size_t build(distance_function_type dfunc, const int k,
                    const knn_index_type& initial_index, const double rho = 0.5,
                    const double delta = 0.001, const bool recheck = false,
                    const double time_limit_sec = 0) {
    typename nn_kernel_type::option option{.k              = k,
                                           .r              = rho,
                                           .delta          = delta,
                                           .time_limit_sec = time_limit_sec,
                                           .exchange_reverse_neighbors = true,
                                           .mini_batch_size = 1 << 26,
                                           .rnd_seed        = m_rnd_seed,
                                           .verbose         = m_verbose};

    nn_kernel_type kernel(option, *m_pstore, priv_get_point_partitioner(),
                          dfunc, m_comm);
    m_knn_index_list->emplace_back();
    kernel.construct(initial_index, recheck, m_knn_index_list->back());
    m_index_k_list->push_back(k);

    return m_knn_index_list->size() - 1;
  }

  /// \brief Build a KNNG.
  /// All ranks must call this function.
  /// \param distance_func_id Distance function ID.
  /// \param k Number of neighbors per point.
  /// \param initial_index Initial index.
  /// \param rho Rho parameter in NN-Descent.
  /// \param delta Delta parameter in NN-Descent.
  /// \param recheck If true, redo the neighbor check for the initial index,
  /// i.e., mark the initial neighbors as 'new' neighbors.
  /// \param time_limit_sec Timeout in seconds for the main neighbor check
  /// kernel. The elapsed time is checked after each neighbor check loop. If the
  /// time limit is exceeded, the construction stops. All ranks must use the
  /// same value. If 0 is given, there is no timeout.
  std::size_t build(
      const distance::id& distance_func_id, const int k,
      const std::unordered_map<id_type, std::vector<id_type>>& initial_index,
      const double rho = 0.5, const double delta = 0.001,
      const bool recheck = false, const double time_limit_sec = 0) {
    return build(distance::distance_function<point_type, distance_type>(
                     distance_func_id),
                 k, initial_index, rho, delta, recheck, time_limit_sec);
  }

  /// \brief Build a KNNG.
  /// All ranks must call this function.
  /// \param dfunc Distance function.
  /// \param k Number of neighbors per point.
  /// \param initial_index Initial index.
  /// \param rho Rho parameter in NN-Descent.
  /// \param delta Delta parameter in NN-Descent.
  /// \param recheck If true, redo the neighbor check for the initial index,
  /// i.e., mark the initial neighbors as 'new' neighbors.
  /// \param time_limit_sec Timeout in seconds for the main neighbor check
  /// kernel. The elapsed time is checked after each neighbor check loop. If the
  /// time limit is exceeded, the construction stops. All ranks must use the
  /// same value. If 0 is given, there is no timeout.
  std::size_t build(
      distance_function_type dfunc, const int k,
      const std::unordered_map<id_type, std::vector<id_type>>& initial_index,
      const double rho = 0.5, const double delta = 0.001,
      const bool recheck = false, const double time_limit_sec = 0) {
    typename nn_kernel_type::option option{.k                          = k,
                                           .r                          = rho,
                                           .delta                      = delta,
                                           .exchange_reverse_neighbors = true,
                                           .mini_batch_size = 1 << 26,
                                           .time_limit_sec  = time_limit_sec,
                                           .rnd_seed        = m_rnd_seed,
                                           .verbose         = m_verbose};

    nn_kernel_type kernel(option, *m_pstore, priv_get_point_partitioner(),
                          dfunc, m_comm);
    m_knn_index_list->emplace_back();
    kernel.construct(initial_index, recheck, m_knn_index_list->back());
    m_index_k_list->push_back(k);

    return m_knn_index_list->size() - 1;
  }

  /// \brief Update the KNNG.
  /// All ranks must call this function.
  void update(const std::size_t index_id, const distance::id& distance_func_id,
              const int k, const double rho = 0.5, const double delta = 0.001,
              const double time_limit_sec = 0) {
    update(index_id,
           distance::distance_function<point_type, distance_type>(
               distance_func_id),
           k, rho, delta, time_limit_sec);
  }

  /// \brief Update the KNNG.
  /// All ranks must call this function.
  void update(const std::size_t index_id, distance_function_type dfunc,
              const int k, const double rho = 0.5, const double delta = 0.001,
              const double time_limit_sec = 0) {
    typename nn_kernel_type::option option{.k              = k,
                                           .r              = rho,
                                           .delta          = delta,
                                           .time_limit_sec = time_limit_sec,
                                           .exchange_reverse_neighbors = true,
                                           .mini_batch_size = 1 << 26,
                                           .rnd_seed        = m_rnd_seed,
                                           .verbose         = m_verbose};

    nn_kernel_type kernel(option, *m_pstore, priv_get_point_partitioner(),
                          dfunc, m_comm);
    kernel.update(m_knn_index_list->at(index_id));
    m_index_k_list->at(index_id) = k;
  }

  /// \brief Apply optimizations to an already constructed KNNG aiming at
  /// improving the query quality and performance.
  /// All ranks must call this function.
  /// \param index_id Index ID.
  /// \param distance_func_id Distance function ID.
  /// \param make_index_undirected If true, make the index undirected.
  /// \param make_index_undirected If true, make the graph undirected.
  /// \param pruning_degree_multiplier
  /// Each point keeps up to k * pruning_degree_multiplier nearest neighbors,
  /// where k is the number of neighbors each point in the index has.
  /// if this value is less than 0, there is no pruning.
  void optimize(const std::size_t   index_id,
                const distance::id& distance_func_id,
                const bool          make_index_undirected     = true,
                const double        pruning_degree_multiplier = 1.5) {
    optimize(index_id,
             distance::distance_function<point_type, distance_type>(
                 distance_func_id),
             make_index_undirected, pruning_degree_multiplier);
  }

  /// \brief Apply optimizations to an already constructed KNNG aiming at
  /// improving the query quality and performance.
  /// All ranks must call this function.
  /// \param index_id Index ID.
  /// \param distance_function Distance function.
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

  template <typename query_iterator>
  std::pair<neighbor_store_type, std::vector<std::vector<point_type>>>
  query_with_features(const std::size_t      index_id,
                      distance_function_type distance_function,
                      query_iterator queries_begin, query_iterator queries_end,
                      const int k, const double epsilon = 0.1) {
    auto query_result = query(index_id, distance_function, queries_begin,
                              queries_end, k, epsilon);
    return std::make_pair(query_result,
                          priv_get_features_for_query_results(query_result));
  }

  /// \brief The same as query() but with distance function ID.
  template <typename query_iterator>
  neighbor_store_type query(const std::size_t   index_id,
                            const distance::id& distance_func_id,
                            query_iterator      queries_begin,
                            query_iterator queries_end, const int k,
                            const double epsilon = 0.1) {
    return query(index_id,
                 distance::distance_function<point_type, distance_type>(
                     distance_func_id),
                 queries_begin, queries_end, k, epsilon);
  }

  /// \brief The same as query_with_features() but with distance function ID.
  template <typename query_iterator>
  std::pair<neighbor_store_type, std::vector<std::vector<point_type>>>
  query_with_features(const std::size_t   index_id,
                      const distance::id& distance_func_id,
                      query_iterator queries_begin, query_iterator queries_end,
                      const int k, const double epsilon = 0.1) {
    return query_with_features(
        index_id,
        distance::distance_function<point_type, distance_type>(
            distance_func_id),
        queries_begin, queries_end, k, epsilon);
  }

  /// \brief Query nearest neighbors of given points.
  /// This function runs queries on multiple indices in such a way that the
  /// indices are merged before the queries are run.
  /// All ranks must call this function.
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

    knn_index_type tmp_index;
    for (auto index_id = index_ids_begin; index_id != index_ids_end;
         ++index_id) {
      tmp_index.merge(m_knn_index_list->at(*index_id));
    }

    query_kernel_type kernel(option, *m_pstore, priv_get_point_partitioner(),
                             distance_function, tmp_index, m_comm);

    query_store_type    queries(queries_begin, queries_end);
    neighbor_store_type query_result;
    kernel.query_batch(queries, query_result);

    return query_result;
  }

  /// \brief The same as query() but runs on multiple indices and returns
  /// neighbor features.
  template <typename index_id_iterator, typename query_iterator>
  std::pair<neighbor_store_type, std::vector<std::vector<point_type>>>
  query_with_features(index_id_iterator      index_ids_begin,
                      index_id_iterator      index_ids_end,
                      distance_function_type distance_function,
                      query_iterator queries_begin, query_iterator queries_end,
                      const int k, const double epsilon = 0.1) {
    auto query_result = query(index_ids_begin, index_ids_end, distance_function,
                              queries_begin, queries_end, k, epsilon);
    return std::make_pair(query_result,
                          priv_get_features_for_query_results(query_result));
  }

  /// \brief The same as query() but with distance function ID and on multiple
  /// indices.
  template <typename index_id_iterator, typename query_iterator>
  neighbor_store_type query(index_id_iterator   index_ids_begin,
                            index_id_iterator   index_ids_end,
                            const distance::id& distance_func_id,
                            query_iterator      queries_begin,
                            query_iterator queries_end, const int k,
                            const double epsilon = 0.1) {
    return query(index_ids_begin, index_ids_end,
                 distance::distance_function<point_type, distance_type>(
                     distance_func_id),
                 queries_begin, queries_end, k, epsilon);
  }

  /// \brief The same as query() but with distance function ID and on multiple
  /// indices. Returns neighbor features also.
  template <typename index_id_iterator, typename query_iterator>
  std::pair<neighbor_store_type, std::vector<std::vector<point_type>>>
  query_with_features(index_id_iterator   index_ids_begin,
                      index_id_iterator   index_ids_end,
                      const distance::id& distance_func_id,
                      query_iterator queries_begin, query_iterator queries_end,
                      const int k, const double epsilon = 0.1) {
    return query(index_ids_begin, index_ids_end,
                 distance::distance_function<point_type, distance_type>(
                     distance_func_id),
                 queries_begin, queries_end, k, epsilon);
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
  void dump_index(const std::size_t index_id, const std::filesystem::path& path,
                  const bool dump_distance = false) const {
    std::stringstream file_name;
    file_name << path.string() << "-" << m_comm.rank();
    const auto ret =
        m_knn_index_list->at(index_id).dump(file_name.str(), dump_distance);
  }

  /// \brief Deprecated API. Use dump_index() instead.
  /// This function will be removed in future releases.
  void dump_graph(const std::size_t index_id, const std::filesystem::path& path,
                  const bool dump_distance = false) const {
    dump_index(index_id, path, dump_distance);
  }

  /// \brief Check if the local point store contains a point with the given ID.
  /// \param id Point ID.
  bool contains_local(const id_type id) const { return m_pstore->contains(id); }

  /// \brief Get the owner rank of a point with the given ID.
  /// \param id Point ID.
  /// \return The rank that owns the point.
  int get_owner(const id_type id) const {
    return priv_get_point_partitioner()(id);
  }

  /// \brief Get a point with the given ID from the local point store.
  const point_type& get_local_point(const id_type id) const {
    return m_pstore->at(id);
  }

  /// \brief Get point data of the given IDs.
  /// This function invokes YGM barrier. All ranks must call this function.
  /// Note: returned data are always stored in normal heap memory, not Metall.
  template <typename id_iterator>
  std::unordered_map<id_type, point_type> get_points(
      id_iterator ids_begin, id_iterator ids_end) const {
    static_assert(
        std::is_same_v<typename std::iterator_traits<id_iterator>::value_type,
                       id_type>,
        "id_iterator must be an iterator of id_type");

    static std::unordered_map<id_type, point_type> return_points_store;
    return_points_store = decltype(return_points_store){};
    return_points_store.reserve(std::distance(ids_begin, ids_end));

    auto proc = [](auto comm, auto pthis, const id_type id,
                   const int source_rank) {
      assert(pthis->contains_local(id));

      // TODO: investigate why point instance is copied in YGM::async() (before
      // sending apparently).
      // Because of that, segmentation fault occurs when
      // Metall is opened as read-only. The below is a workaround to avoid the
      // issue: Allocate point instance in the heap, not in Metall, explicitly.
      // If fallback_allocator is used, constructing the point_type without
      // allocator instance falls back to the default (heap) allocator.
      point_type point;
      // TODO: check the value of propagate_on_container_copy_assignment.
      // Assumes that the propagate_on_container_copy_assignment of the
      // allocator_type of the point_type is false.
      point = pthis->get_local_point(id);

      comm->async(
          source_rank,
          [](auto, const auto& id, auto point) {
            // Avoid duplicate insertion to get better performance.
            if (!return_points_store.contains(id)) {
              return_points_store.emplace(id, std::move(point));
            }
          },
          id, std::move(point));
    };
    m_comm.cf_barrier();

    for (auto it = ids_begin; it != ids_end; ++it) {
      const auto id = *it;
      m_comm.async(get_owner(id), proc, m_this, id, m_comm.rank());
    }
    m_comm.barrier();

    return return_points_store;
  }

  /// \brife Returns an iterator that points to the beginning of the locally
  /// stored points.
  auto local_points_begin() const { return m_pstore->begin(); }

  /// \brief Returns an iterator that points to the end of the locally stored
  /// points.
  auto local_points_end() const { return m_pstore->end(); }

  /// \brief Get the number of locally stored points.
  std::size_t num_local_points() const { return m_pstore->size(); }

  /// \brief Get the number of points.
  /// This function performs an all-reduce operation, which is not cheap.
  std::size_t num_points() const { return ygm::sum(m_pstore->size(), m_comm); }

  /// \brief API for using 'for_each' with local points.
  iterator_proxy_type local_points() const {
    return iterator_proxy_type(local_points_begin(), local_points_end());
  }

  /// \brief Erase a kNN index
  /// \param index_id Index ID.
  void erase(const std::size_t index_id) {
    m_knn_index_list->erase(m_knn_index_list->begin() + index_id);
    m_index_k_list->erase(m_index_k_list->begin() + index_id);
  }

  /// \brief Get the number of neighbors of the given point.
  /// If the point is not stored locally, the function returns 0
  /// \param index_id Index ID.
  /// \param id Point ID.
  /// \return The number of neighbors of the point.
  std::size_t num_local_neighbors(std::size_t   index_id,
                                  const id_type id) const {
    if (contains_local(id)) {
      return m_knn_index_list->at(index_id).at(id).size();
    }
    return 0;
  }

  /// \brief Get the neighbors of the given local point.
  /// If the point is not stored locally, the function throws an exception.
  /// \param id Point ID.
  /// \return The neighbors of the point. A vector of neighbor.
  std::vector<neighbor_type> get_local_neighbors(std::size_t   index_id,
                                                 const id_type id) const {
    std::vector<neighbor_type> neighbors;
    for (auto itr = m_knn_index_list->at(index_id).neighbors_begin(id),
              end = m_knn_index_list->at(index_id).neighbors_end(id);
         itr != end; ++itr) {
      neighbors.push_back(*itr);
    }
    return neighbors;
  }

  /// \brief Get the neighbors of the given point.
  /// This function invokes YGM barrier. All ranks must call this function.
  template <typename id_iterator>
  std::unordered_map<id_type, std::vector<neighbor_type>> get_neighbors(
      std::size_t index_id, id_iterator ids_begin, id_iterator ids_end) const {
    static_assert(
        std::is_same_v<typename std::iterator_traits<id_iterator>::value_type,
                       id_type>,
        "id_iterator must be an iterator of id_type");

    static std::unordered_map<id_type, std::vector<neighbor_type>>
        neighbors_table;
    neighbors_table = decltype(neighbors_table){};
    neighbors_table.reserve(std::distance(ids_begin, ids_end));

    auto proc = [](auto comm, auto pthis, const std::size_t index_id,
                   const id_type id, const int source_rank) {
      assert(pthis->contains_local(id));
      const auto neighbors = pthis->get_local_neighbors(index_id, id);
      comm->async(
          source_rank,
          [](auto, const auto& id, const auto& neighbors) {
            neighbors_table.emplace(id, neighbors);
          },
          id, neighbors);
    };
    m_comm.cf_barrier();

    for (auto it = ids_begin; it != ids_end; ++it) {
      const auto id = *it;
      m_comm.async(get_owner(id), proc, m_this, index_id, id, m_comm.rank());
    }
    m_comm.barrier();

    return neighbors_table;
  }

  /// \brief Get the neighbors of the given point with features of the
  /// neighbors.
  template <typename id_iterator>
  std::unordered_map<
      id_type, std::pair<std::vector<neighbor_type>, std::vector<point_type>>>
  get_neighbors_with_features(std::size_t index_id, id_iterator ids_begin,
                              id_iterator ids_end) const {
    // Get neighbors
    const auto neighbors_table = get_neighbors(index_id, ids_begin, ids_end);

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
      // Construct neighbor feature vector
      // The line below makes the fallback_allocator in the point_type not to
      // get stateful allocator.
      std::vector<point_type> neighbor_features(neighbors.size());
      for (size_t i = 0; i < neighbors.size(); ++i) {
        neighbor_features[i] = neighbor_features_table.at(neighbors[i].id);
      }

      result[id] =
          std::make_pair(std::move(neighbors), std::move(neighbor_features));
    }
    m_comm.cf_barrier();

    return result;
  }

  std::vector<std::size_t> get_index_ids() const {
    std::vector<std::size_t> index_ids;
    for (std::size_t i = 0; i < m_knn_index_list->size(); ++i) {
      index_ids.push_back(i);
    }
    return index_ids;
  }

  /// \brief Return the reference to k-NN index associated with index_id.
  /// \param index_id Index ID.
  const knn_index_type& get_index(const std::size_t index_id) const {
    // Stable vector (or similar container) must be used to avoid dangling
    // reference when m_knn_index_list's size is changed.
    return m_knn_index_list->at(index_id);
  }

  /// \brief Create a snapshot of the current persistent datastore.
  /// \param dest_datastore_path Destination path of the snapshot.
  /// \return True if the snapshot is successfully created; false otherwise.
  /// \Note This function cannot be called when Metall is not used.
  /// This function cannot be called concurrently with other write functions,
  /// such as add_points() and load_points().
  bool snapshot(const std::filesystem::path& dest_datastore_path) {
    if (!m_metall) {
      m_comm.cerr0() << "Error: copy_pm_datastore() cannot be called when "
                        "Metall is not used."
                     << std::endl;
      return false;
    }

    const auto ret = m_metall->snapshot(dest_datastore_path, true);
    if (!ret) {
      m_comm.cerr0() << "Error: Failed to create a snapshot to "
                     << dest_datastore_path << std::endl;
      return false;
    }
    if (m_verbose) {
      m_comm.cout0() << "A snapshot is created at " << dest_datastore_path
                     << std::endl;
    }
    return true;
  }

 private:
  /// \brief Return a point partitioner instance.
  /// \return A point partitioner instance.
  point_partitioner priv_get_point_partitioner() const {
    const int size = m_comm.size();
    return [size](const id_type& id) { return get_owner(id, size); };
  };

  std::vector<std::vector<point_type>> priv_get_features_for_query_results(
      const neighbor_store_type& query_results) {
    std::set<id_type> neighbor_ids;
    for (const auto& neighbors : query_results) {
      for (const auto& neighbor : neighbors) {
        neighbor_ids.insert(neighbor.id);
      }
    }
    auto neighbor_features_table =
        get_points(neighbor_ids.begin(), neighbor_ids.end());

    std::vector<std::vector<point_type>> neighbor_features_to_return;
    for (std::size_t i = 0; i < query_results.size(); ++i) {
      std::vector<point_type> neighbor_features(query_results[i].size());
      for (std::size_t j = 0; j < query_results[i].size(); ++j) {
        neighbor_features[j] =
            neighbor_features_table.at(query_results[i][j].id);
      }
      neighbor_features_to_return.push_back(std::move(neighbor_features));
    }

    return neighbor_features_to_return;
  }

  ygm::comm&                                           m_comm;
  uint64_t                                             m_rnd_seed;
  bool                                                 m_verbose;
  std::unique_ptr<metall::utility::metall_mpi_adaptor> m_metall{nullptr};
  std::unique_ptr<point_store_type>                    m_pstore{nullptr};
  std::unique_ptr<knn_index_container> m_knn_index_list{nullptr};
  std::unique_ptr<size_container>      m_index_k_list{nullptr};
  ygm::ygm_ptr<self_type>              m_this{this};
};

}  // namespace saltatlas