// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <fstream>
#ifdef SALTATLAS_DNND_INCLUDED_HPP
#error \
    "saltatlas/dnnd/dnnd.hpp is already included. Please include either saltatlas/dnnd/dnnd.hpp or saltatlas/dnnd/dnnd_adv.hpp, but not both."
#endif

#ifndef SALTATLAS_DNND_ADV_INCLUDED_HPP
#define SALTATLAS_DNND_ADV_INCLUDED_HPP
#endif  // SALTATLAS_DNND_ADV_INCLUDED_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/interprocess/containers/stable_vector.hpp>
#include <boost/version.hpp>
#if defined(BOOST_VERSION) && BOOST_VERSION >= 108700
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#else
#error "Boost 1.87.00 or higher is required."
#endif

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
#include "saltatlas/dnnd/utility.hpp"

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

/// \brief Distributed NNDescent advanced version.
/// \tparam Id Point ID type.
/// \tparam Point Point type.
/// \tparam Distance Distance type.
/// \tparam IdHash Point ID hasher for externally keyed APIs.
template <typename Id       = uint64_t,
          typename Point    = saltatlas::pm_feature_vector<double>,
          typename Distance = double, typename IdHash = std::hash<Id>>
class dnnd_adv {
 private:
  using self_type = dnnd_adv<Id, Point, Distance, IdHash>;
  using mmanager  = metall::utility::metall_mpi_adaptor::manager_type;

  constexpr static unsigned int k_pstore_hash_seed            = 0xA1B2C3D4;
  constexpr static unsigned int k_point_partitioner_hash_seed = 0x1A2B3C4D;
  static_assert(k_pstore_hash_seed != k_point_partitioner_hash_seed,
                "k_pstore_hash_seed and k_point_partitioner_hash_seed must be "
                "different.");

  static constexpr const char* k_e2i_id_table_name = "dnnd_adv_e2i_id_table";
  static constexpr const char* k_local_e2i_id_table_name =
      "dnnd_adv_local_e2i_id_table";
  static constexpr const char* k_i2e_id_table_name = "dnnd_adv_i2e_id_table";

  template <typename T>
  using allocator_type = mmanager::fallback_allocator<T>;

  template <typename T>
  using scp_allocator_type = mmanager::scoped_fallback_allocator_type<T>;

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
  /// as the internal ID.
  static constexpr bool k_use_eid_table = !std::is_integral_v<id_type>;

  /// \brief Internal ID hasher for point store.
  using pstore_internal_id_hasher = hash<k_pstore_hash_seed>;

  /// \brief Internal ID hasher for point partitioner.
  using point_partitioner_internal_id_hasher =
      hash<k_point_partitioner_hash_seed>;

  /// \brief External ID hasher for point store.
  struct external_id_pstore_hasher {
    inline std::size_t operator()(const id_type& id) const {
      return hash<k_pstore_hash_seed>{}(hasher{}(id));
    }
  };

  /// \brief External ID hasher for point partitioner.
  struct external_id_partitioner_hasher {
    inline std::size_t operator()(const id_type& id) const {
      return hash<k_point_partitioner_hash_seed>{}(hasher{}(id));
    }
  };

  /// \brief Point store type.
  using point_store_type =
      point_store<internal_id_type, point_type, pstore_internal_id_hasher,
                  std::equal_to<>, allocator_type<std::byte>>;
  using external_point_store_type =
      point_store<id_type, point_type, external_id_pstore_hasher,
                  std::equal_to<>, allocator_type<std::byte>>;

 public:
  /// \brief k-NN index type with external ID type (i.e., id_type).
  using knn_index_type = dndetail::nn_index<id_type, distance_type,
                                            allocator_type<std::byte>, hasher>;

 private:
  /// \brief k-NN index type.
  using internal_knn_index_type =
      dndetail::nn_index<internal_id_type, distance_type,
                         allocator_type<std::byte>>;

  using nn_kernel_type = dndetail::dnnd_kernel<point_store_type, distance_type>;

  /// \brief Point partitioner type.
  using internal_point_partitioner = typename nn_kernel_type::point_partitioner;

  using nn_index_optimizer_type =
      dndetail::nn_index_optimizer<point_store_type, internal_knn_index_type>;

  using query_kernel_type =
      dndetail::dknn_batch_query_kernel<point_store_type,
                                        internal_knn_index_type>;

  using query_store_type = typename query_kernel_type::query_store_type;
  using internal_neighbor_store_type =
      typename query_kernel_type::neighbor_store_type;

  using internal_knn_index_container = boost::interprocess::stable_vector<
      internal_knn_index_type, scp_allocator_type<internal_knn_index_type>>;
  using size_container =
      metall::container::vector<std::size_t, scp_allocator_type<std::size_t>>;

  using e2i_id_table_type = boost::unordered::unordered_flat_map<
      id_type, internal_id_type, hasher, std::equal_to<>,
      scp_allocator_type<std::pair<const id_type, internal_id_type>>>;

  using i2e_id_table_type = boost::unordered::unordered_flat_map<
      internal_id_type, id_type, pstore_internal_id_hasher, std::equal_to<>,
      scp_allocator_type<std::pair<const internal_id_type, id_type>>>;

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

  using external_initial_index_type =
      std::unordered_map<id_type, std::vector<id_type>,
                         external_id_pstore_hasher>;

  static constexpr int get_owner(const id_type& id, const int mpi_size) {
    static_assert(!k_use_eid_table,
                  "get_owner() is not available when external ID and internal "
                  "ID are different.");
    return point_partitioner_internal_id_hasher{}(id) % mpi_size;
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
    m_knn_index_list = std::make_unique<internal_knn_index_container>();
    m_index_k_list   = std::make_unique<size_container>();
    priv_init_dram_id_tables();
    m_comm.cf_barrier();
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
    m_knn_index_list.reset(localm.construct<internal_knn_index_container>(
        metall::unique_instance)(localm.get_allocator<>()));
    m_index_k_list.reset(localm.construct<size_container>(
        metall::unique_instance)(localm.get_allocator<>()));
    priv_construct_persistent_id_tables(localm);
    m_comm.cf_barrier();
  }

  /// \brief Constructor. This constructor opens an existing persistent
  /// (Metall) datastore.
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
        localm.find<internal_knn_index_container>(metall::unique_instance)
            .first);
    assert(m_knn_index_list);
    m_index_k_list.reset(
        localm.find<size_container>(metall::unique_instance).first);
    assert(m_index_k_list);
    priv_open_persistent_id_tables(localm);
    m_comm.cf_barrier();
  }

  /// \brief Constructor. This constructor opens an existing persistent
  /// (Metall) datastore in read-only mode.
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
        localm.find<internal_knn_index_container>(metall::unique_instance)
            .first);
    assert(m_knn_index_list);
    m_index_k_list.reset(
        localm.find<size_container>(metall::unique_instance).first);
    assert(m_index_k_list);
    priv_open_persistent_id_tables(localm);
    m_comm.cf_barrier();
  }

  ~dnnd_adv() noexcept {
    if (m_metall) {
      // To keep the objects in the Metall datastore, do not destroy them.
      m_pstore.release();
      m_knn_index_list.release();
      m_index_k_list.release();
      if constexpr (k_use_eid_table) {
        m_e2i_id_table.release();
        m_local_e2i_id_table.release();
        m_i2e_id_table.release();
      }
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
      m_comm.cout() << "Contains " << m_pstore->size()
                    << " points after adding points." << std::endl;
      m_comm.cf_barrier();
    }
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

    std::vector<std::filesystem::path> point_file_paths(paths_begin, paths_end);
    if constexpr (k_use_eid_table) {
      external_point_store_type pstore;
      saltatlas::read_points(point_file_paths, file_format, m_verbose,
                             priv_get_point_partitioner_external(), pstore,
                             m_comm);

      std::vector<id_type> eids;
      eids.reserve(pstore.size());
      for (const auto& [eid, _] : pstore) {
        eids.push_back(priv_copy_value_to_heap(eid));
      }
      priv_gen_internal_id(eids.begin(), eids.end());

      for (const auto& [eid, point] : pstore) {
        priv_add_point_async(eid, point);
      }
      m_comm.barrier();
    } else {
      saltatlas::read_points(point_file_paths, file_format, m_verbose,
                             priv_get_point_partitioner_internal(), *m_pstore,
                             m_comm);
    }
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

    if constexpr (k_use_eid_table) {
      external_point_store_type pstore;
      saltatlas::detail::read_points_with_id_helper(
          point_file_paths, parser_wrapper, pstore,
          priv_get_point_partitioner_external(), m_comm, m_verbose);

      std::vector<id_type> eids;
      eids.reserve(pstore.size());
      for (const auto& [eid, _] : pstore) {
        eids.push_back(priv_copy_value_to_heap(eid));
      }
      priv_gen_internal_id(eids.begin(), eids.end());

      for (const auto& [eid, point] : pstore) {
        priv_add_point_async(eid, point);
      }
      m_comm.barrier();
    } else {
      saltatlas::detail::read_points_with_id_helper(
          point_file_paths, parser_wrapper, *m_pstore,
          priv_get_point_partitioner_internal(), m_comm, m_verbose);
    }
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

    nn_kernel_type kernel(option, *m_pstore,
                          priv_get_point_partitioner_internal(), dfunc, m_comm);
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
  /// can be used. Currently, index must be partitioned in the same way as the
  /// point store.
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
  /// can be used. Currently, index must be partitioned in the same way as the
  /// point store.
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

    nn_kernel_type kernel(option, *m_pstore,
                          priv_get_point_partitioner_internal(), dfunc, m_comm);
    m_knn_index_list->emplace_back();
    kernel.construct(initial_index, recheck, m_knn_index_list->back());
    m_index_k_list->push_back(k);

    return m_knn_index_list->size() - 1;
  }

  /// \brief Build a KNNG.
  /// All ranks must call this function.
  /// \param distance_func_id Distance function ID.
  /// \param k Number of neighbors per point.
  /// \param initial_index Initial index. Currently, index must be partitioned
  /// in the same way as the point store.
  /// \param rho Rho parameter in NN-Descent.
  /// \param delta Delta parameter in NN-Descent.
  /// \param recheck If true, redo the neighbor check for the initial index,
  /// i.e., mark the initial neighbors as 'new' neighbors.
  /// \param time_limit_sec Timeout in seconds for the main neighbor check
  /// kernel. The elapsed time is checked after each neighbor check loop. If the
  /// time limit is exceeded, the construction stops. All ranks must use the
  /// same value. If 0 is given, there is no timeout.
  std::size_t build(const distance::id& distance_func_id, const int k,
                    const external_initial_index_type& initial_index,
                    const double rho = 0.5, const double delta = 0.001,
                    const bool   recheck        = false,
                    const double time_limit_sec = 0) {
    return build(distance::distance_function<point_type, distance_type>(
                     distance_func_id),
                 k, initial_index, rho, delta, recheck, time_limit_sec);
  }

  /// \brief Build a KNNG.
  /// All ranks must call this function.
  /// \param dfunc Distance function.
  /// \param k Number of neighbors per point.
  /// \param initial_index Initial index. Currently, index must be partitioned
  /// in the same way as the point store.
  /// \param rho Rho parameter in NN-Descent.
  /// \param delta Delta parameter in NN-Descent.
  /// \param recheck If true, redo the neighbor check for the initial index,
  /// i.e., mark the initial neighbors as 'new' neighbors.
  /// \param time_limit_sec Timeout in seconds for the main neighbor check
  /// kernel. The elapsed time is checked after each neighbor check loop. If the
  /// time limit is exceeded, the construction stops. All ranks must use the
  /// same value. If 0 is given, there is no timeout.
  std::size_t build(distance_function_type dfunc, const int k,
                    const external_initial_index_type& initial_index,
                    const double rho = 0.5, const double delta = 0.001,
                    const bool   recheck        = false,
                    const double time_limit_sec = 0) {
    typename nn_kernel_type::option option{.k              = k,
                                           .r              = rho,
                                           .delta          = delta,
                                           .time_limit_sec = time_limit_sec,
                                           .exchange_reverse_neighbors = true,
                                           .mini_batch_size = 1 << 26,
                                           .rnd_seed        = m_rnd_seed,
                                           .verbose         = m_verbose};

    nn_kernel_type kernel(option, *m_pstore,
                          priv_get_point_partitioner_internal(), dfunc, m_comm);
    m_knn_index_list->emplace_back();
    if constexpr (k_use_eid_table) {
      const auto internal_initial_index =
          priv_gen_internal_initial_index(initial_index);
      kernel.construct(internal_initial_index, recheck,
                       m_knn_index_list->back());
    } else {
      kernel.construct(initial_index, recheck, m_knn_index_list->back());
    }
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

    nn_kernel_type kernel(option, *m_pstore,
                          priv_get_point_partitioner_internal(), dfunc, m_comm);
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
                                      priv_get_point_partitioner_internal(),
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
    const auto query_results =
        priv_run_query(m_knn_index_list->at(index_id), distance_function,
                       queries_begin, queries_end, k, epsilon);
    if constexpr (k_use_eid_table) {
      return query_results.first;
    } else {
      return query_results.second;
    }
  }

  template <typename query_iterator>
  std::pair<neighbor_store_type, std::vector<std::vector<point_type>>>
  query_with_features(const std::size_t      index_id,
                      distance_function_type distance_function,
                      query_iterator queries_begin, query_iterator queries_end,
                      const int k, const double epsilon = 0.1) {
    auto query_results =
        priv_run_query(m_knn_index_list->at(index_id), distance_function,
                       queries_begin, queries_end, k, epsilon);
    auto neighbor_features =
        priv_gather_neighbor_features(query_results.second);

    if constexpr (k_use_eid_table) {
      return std::make_pair(std::move(query_results.first),
                            std::move(neighbor_features));
    } else {
      return std::make_pair(std::move(query_results.second),
                            std::move(neighbor_features));
    }
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
    internal_knn_index_type tmp_index;
    for (auto index_id = index_ids_begin; index_id != index_ids_end;
         ++index_id) {
      tmp_index.merge(m_knn_index_list->at(*index_id));
    }

    const auto query_results = priv_run_query(
        tmp_index, distance_function, queries_begin, queries_end, k, epsilon);
    if constexpr (k_use_eid_table) {
      return query_results.first;
    } else {
      return query_results.second;
    }
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
    internal_knn_index_type tmp_index;
    for (auto index_id = index_ids_begin; index_id != index_ids_end;
         ++index_id) {
      tmp_index.merge(m_knn_index_list->at(*index_id));
    }

    auto query_results = priv_run_query(tmp_index, distance_function,
                                        queries_begin, queries_end, k, epsilon);
    auto neighbor_features =
        priv_gather_neighbor_features(query_results.second);
    if constexpr (k_use_eid_table) {
      return std::make_pair(std::move(query_results.first),
                            std::move(neighbor_features));
    } else {
      return std::make_pair(std::move(query_results.second),
                            std::move(neighbor_features));
    }
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
    return query_with_features(
        index_ids_begin, index_ids_end,
        distance::distance_function<point_type, distance_type>(
            distance_func_id),
        queries_begin, queries_end, k, epsilon);
  }

  /// \brief Dump the k-NN index to distributed files.
  /// \param path File path prefix.
  /// \param dump_distance If true, also dump distances.
  void dump_index(const std::size_t index_id, const std::filesystem::path& path,
                  const bool dump_distance = false) const {
    auto parent_path = path.parent_path();
    if (!parent_path.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(parent_path, ec);
      if (ec) {
        throw std::runtime_error("Failed to create directories: " +
                                 ec.message());
      }
    }

    std::stringstream file_name;
    file_name << path.string() << "-" << m_comm.rank();
    if constexpr (k_use_eid_table) {
      const auto external_index =
          priv_gen_external_knng(m_knn_index_list->at(index_id));
      external_index.dump(file_name.str(), dump_distance);
    } else {
      m_knn_index_list->at(index_id).dump(file_name.str(), dump_distance);
    }
    m_comm.cf_barrier();
  }

  /// \brief Deprecated API. Use dump_index() instead.
  /// This function will be removed in future releases.
  void dump_graph(const std::size_t index_id, const std::filesystem::path& path,
                  const bool dump_distance = false) const {
    dump_index(index_id, path, dump_distance);
  }

  /// \brief Check if the local point store contains a point with the given ID.
  /// \param id Point ID.
  bool contains_local(const id_type id) const {
    if constexpr (k_use_eid_table) {
      return m_local_e2i_id_table->contains(id);
    } else {
      return m_pstore->contains(id);
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

  /// \brief Get a point with the given ID from the local point store.
  const point_type& get_local_point(const id_type id) const {
    return m_pstore->at(priv_find_local_internal_id(id));
  }

  /// \brief Get point data of the given IDs.
  /// This function invokes YGM barrier. All ranks must call this function.
  /// Note: returned data are always stored in normal heap memory, not Metall.
  template <typename id_iterator>
  std::unordered_map<id_type, point_type, external_id_pstore_hasher> get_points(
      id_iterator ids_begin, id_iterator ids_end) const {
    static_assert(
        std::is_same_v<typename std::iterator_traits<id_iterator>::value_type,
                       id_type>,
        "id_iterator must be an iterator of id_type");

    return priv_get_remote_points(ids_begin, ids_end);
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

  /// \brief Erase a kNN index.
  /// \param index_id Index ID.
  void erase(const std::size_t index_id) {
    m_knn_index_list->erase(m_knn_index_list->begin() + index_id);
    m_index_k_list->erase(m_index_k_list->begin() + index_id);
  }

  /// \brief Get the number of neighbors of the given point.
  /// If the point is not stored locally, the function returns 0.
  /// \param index_id Index ID.
  /// \param id Point ID.
  /// \return The number of neighbors of the point.
  std::size_t num_local_neighbors(std::size_t   index_id,
                                  const id_type id) const {
    if (contains_local(id)) {
      return m_knn_index_list->at(index_id).num_neighbors(
          priv_find_local_internal_id(id));
    }
    return 0;
  }

  /// \brief Get the neighbors of the given local point.
  /// If the point is not stored locally, the function throws an exception.
  /// \param id Point ID.
  /// \return The neighbors of the point. A vector of neighbor.
  std::vector<neighbor_type> get_local_neighbors(std::size_t   index_id,
                                                 const id_type id) const {
    if constexpr (k_use_eid_table) {
      return priv_get_local_external_neighbors(m_knn_index_list->at(index_id),
                                               priv_find_local_internal_id(id));
    } else {
      std::vector<neighbor_type> neighbors;
      for (auto itr = m_knn_index_list->at(index_id).neighbors_begin(id),
                end = m_knn_index_list->at(index_id).neighbors_end(id);
           itr != end; ++itr) {
        neighbors.push_back(*itr);
      }
      return neighbors;
    }
  }

  /// \brief Get the neighbors of the given point.
  /// This function invokes YGM barrier. All ranks must call this function.
  template <typename id_iterator>
  std::unordered_map<id_type, std::vector<neighbor_type>,
                     external_id_pstore_hasher>
  get_neighbors(std::size_t index_id, id_iterator ids_begin,
                id_iterator ids_end) const {
    static_assert(
        std::is_same_v<typename std::iterator_traits<id_iterator>::value_type,
                       id_type>,
        "id_iterator must be an iterator of id_type");

    static std::unordered_map<id_type, std::vector<neighbor_type>,
                              external_id_pstore_hasher>
        neighbors_table;
    neighbors_table = decltype(neighbors_table){};
    neighbors_table.reserve(std::distance(ids_begin, ids_end));

    using internal_neighbor_type =
        typename internal_knn_index_type::neighbor_type;
    static std::unordered_map<id_type, std::vector<internal_neighbor_type>,
                              external_id_pstore_hasher>
        internal_neighbors_table;
    if constexpr (k_use_eid_table) {
      internal_neighbors_table = decltype(internal_neighbors_table){};
      internal_neighbors_table.reserve(std::distance(ids_begin, ids_end));
    }

    m_comm.cf_barrier();
    for (auto it = ids_begin; it != ids_end; ++it) {
      const auto id = *it;
      if constexpr (k_use_eid_table) {
        m_comm.async(
            priv_get_point_partitioner_external()(id),
            [](auto comm, auto pthis, const std::size_t index_id,
               const id_type eid, const int source_rank) {
              const auto iid = pthis->priv_find_local_internal_id(eid);
              comm->async(
                  pthis->priv_get_point_partitioner_internal()(iid),
                  [](auto comm, auto pthis, const std::size_t index_id,
                     const id_type eid, const internal_id_type iid,
                     const int source_rank) {
                    assert(pthis->m_pstore->contains(iid));
                    std::vector<internal_neighbor_type> neighbors;
                    for (auto itr = pthis->m_knn_index_list->at(index_id)
                                        .neighbors_begin(iid),
                              end = pthis->m_knn_index_list->at(index_id)
                                        .neighbors_end(iid);
                         itr != end; ++itr) {
                      neighbors.push_back(*itr);
                    }
                    comm->async(
                        source_rank,
                        [](auto, const auto& eid, const auto& neighbors) {
                          internal_neighbors_table.emplace(eid, neighbors);
                        },
                        pthis->priv_copy_value_to_heap(eid), neighbors);
                  },
                  pthis->m_this, index_id, pthis->priv_copy_value_to_heap(eid),
                  iid, source_rank);
            },
            m_this, index_id, id, m_comm.rank());
      } else {
        m_comm.async(
            priv_get_point_partitioner_internal()(id),
            [](auto comm, auto pthis, const std::size_t index_id,
               const id_type id, const int source_rank) {
              assert(pthis->m_pstore->contains(id));
              const auto neighbors = pthis->get_local_neighbors(index_id, id);
              comm->async(
                  source_rank,
                  [](auto, const auto& id, const auto& neighbors) {
                    neighbors_table.emplace(id, neighbors);
                  },
                  id, neighbors);
            },
            m_this, index_id, id, m_comm.rank());
      }
    }
    m_comm.barrier();

    if constexpr (k_use_eid_table) {
      std::set<internal_id_type> neighbor_ids;
      for (const auto& [eid, neighbors] : internal_neighbors_table) {
        (void)eid;
        for (const auto& neighbor : neighbors) {
          neighbor_ids.insert(neighbor.id);
        }
      }

      const auto i2e_id_map =
          priv_get_external_ids(neighbor_ids.begin(), neighbor_ids.end());
      for (const auto& [eid, neighbors] : internal_neighbors_table) {
        auto& external_neighbors = neighbors_table[eid];
        external_neighbors.reserve(neighbors.size());
        for (const auto& neighbor : neighbors) {
          external_neighbors.emplace_back(
              priv_copy_value_to_heap(i2e_id_map.at(neighbor.id)),
              neighbor.distance);
        }
      }
    }

    return neighbors_table;
  }

  /// \brief Get the neighbors of the given point with features of the
  /// neighbors.
  template <typename id_iterator>
  std::unordered_map<
      id_type, std::pair<std::vector<neighbor_type>, std::vector<point_type>>,
      external_id_pstore_hasher>
  get_neighbors_with_features(std::size_t index_id, id_iterator ids_begin,
                              id_iterator ids_end) const {
    const auto neighbors_table = get_neighbors(index_id, ids_begin, ids_end);

    std::set<id_type> neighbor_ids;
    for (const auto& [id, neighbors] : neighbors_table) {
      for (const auto& neighbor : neighbors) {
        neighbor_ids.insert(neighbor.id);
      }
    }
    const auto neighbor_features_table =
        get_points(neighbor_ids.begin(), neighbor_ids.end());

    std::unordered_map<
        id_type, std::pair<std::vector<neighbor_type>, std::vector<point_type>>,
        external_id_pstore_hasher>
        result;
    for (auto& [id, neighbors] : neighbors_table) {
      std::vector<point_type> neighbor_features(neighbors.size());
      for (std::size_t i = 0; i < neighbors.size(); ++i) {
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

  /// \brief Return the k-NN index associated with index_id.
  /// \param index_id Index ID.
  decltype(auto) get_index(const std::size_t index_id) const {
    if constexpr (k_use_eid_table) {
      return priv_gen_external_knng(m_knn_index_list->at(index_id));
    } else {
      // Stable vector (or similar container) must be used to avoid dangling
      // reference when m_knn_index_list's size is changed.
      return (m_knn_index_list->at(index_id));
    }
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

  // Dump the external ID to internal ID mapping table to a single file.
  // There are two columns: external ID and internal ID.
  bool dump_external_id_map(const std::filesystem::path& output_path) const {
    if constexpr (!k_use_eid_table) {
      m_comm.cerr0() << "Error: dump_external_id_map() cannot be called when "
                        "external ID and internal ID are the same."
                     << std::endl;
      return false;
    }

    std::ofstream ofs;
    if (m_comm.rank0()) {
      ofs.open(output_path);
      if (!ofs) {
        m_comm.cerr0() << "Error: Failed to open file " << output_path
                       << " for writing." << std::endl;
        return false;
      }
      // ofs << "external_id internal_id\n";
    }
    ygm::ygm_ptr<std::ofstream> ptr_ofs{&ofs};
    m_comm.cf_barrier();

    for (const auto& [eid, iid] : *m_e2i_id_table) {
      m_comm.async(
          0,
          [](auto comm, auto ptr_ofs, const id_type eid,
             const internal_id_type iid) {
            (*ptr_ofs) << eid << " " << iid << "\n";
          },
          ptr_ofs, eid, iid);
    }
    m_comm.barrier();

    return true;
  }

  void aggregate_datastore(const std::filesystem::path& output_datastore_path) {
    std::shared_ptr<metall::manager>           ptr_main_manager{nullptr};
    ygm::ygm_ptr<point_store_type>             ptr_main_pstore{nullptr};
    ygm::ygm_ptr<internal_knn_index_container> ptr_main_knng_index_list{
        nullptr};

    ygm::ygm_ptr<e2i_id_table_type> ptr_main_e2i_id_table{nullptr};
    ygm::ygm_ptr<i2e_id_table_type> ptr_main_i2e_id_table{nullptr};

    if (m_comm.rank0()) {
      // Create a new Metall manager for the aggregated datastore.
      ptr_main_manager.reset(
          new metall::manager(metall::create_only, output_datastore_path));
      ptr_main_pstore = ptr_main_manager->construct<point_store_type>(
          metall::unique_instance)();
      ptr_main_knng_index_list =
          ptr_main_manager->construct<internal_knn_index_container>(
              metall::unique_instance)();

      if constexpr (k_use_eid_table) {
        ptr_main_e2i_id_table = ptr_main_manager->construct<e2i_id_table_type>(
            k_e2i_id_table_name)(ptr_main_manager->get_allocator<>());
        ptr_main_i2e_id_table = ptr_main_manager->construct<i2e_id_table_type>(
            k_i2e_id_table_name)(ptr_main_manager->get_allocator<>());
      }

      auto index_k_list = ptr_main_manager->construct<size_container>(
          metall::unique_instance)();
      *index_k_list = *m_index_k_list;
    }
    m_comm.cf_barrier();

    // Assume that local data is already constructed or opened.
    for (auto itr = m_pstore->begin(); itr != m_pstore->end(); ++itr) {
      const auto sid   = itr->first;
      const auto point = itr->second;
      m_comm.async(
          0,
          [](auto comm, auto ptr_main_pstore, const internal_id_type sid,
             const point_type& point) {
            (*ptr_main_pstore)[sid] = priv_copy_value_to_heap(point);
          },
          ptr_main_pstore, sid, point);
    }

    // Next, merge local k-NN indices into a single index on rank 0.
    // Send each element one by one to rank 0.
    for (std::size_t index_id = 0; index_id < m_knn_index_list->size();
         ++index_id) {
      for (auto itr = m_knn_index_list->at(index_id).begin();
           itr != m_knn_index_list->at(index_id).end(); ++itr) {
        const auto id        = itr->first;
        const auto neighbors = itr->second;
        m_comm.async(
            0,
            [](auto comm, auto ptr_main_knng_index_list,
               const std::size_t index_id, const internal_id_type id,
               const auto& neighbors) {
              // If the index_id is greater than the current size of the main
              // index list, emplace a new index to the main index list.
              if (index_id >= ptr_main_knng_index_list->size()) {
                ptr_main_knng_index_list->emplace_back();
              }
              // Add neighbors to the main k-NN index list.
              auto& main_index = ptr_main_knng_index_list->at(index_id);
              for (const auto& neighbor : neighbors) {
                main_index.insert(id, neighbor);
              }
            },
            ptr_main_knng_index_list, index_id, id, neighbors);
      }
    }

    // Next, merge ID mapping tables if they exist.
    if constexpr (k_use_eid_table) {
      for (auto itr = m_e2i_id_table->begin(); itr != m_e2i_id_table->end();
           ++itr) {
        const auto eid = itr->first;
        const auto iid = itr->second;
        m_comm.async(
            0,
            [](auto comm, auto ptr_main_e2i_id_table,
               auto ptr_main_i2e_id_table, const id_type eid,
               const internal_id_type iid) {
              (*ptr_main_e2i_id_table)[eid] = iid;
              (*ptr_main_i2e_id_table)[iid] = eid;
            },
            ptr_main_e2i_id_table, ptr_main_i2e_id_table, eid, iid);
      }
    }

    // Wait for all async operations to finish.
    m_comm.barrier();
  }

 private:
  void priv_init_dram_id_tables() {
    if constexpr (k_use_eid_table) {
      m_e2i_id_table       = std::make_unique<e2i_id_table_type>();
      m_local_e2i_id_table = std::make_unique<e2i_id_table_type>();
      m_i2e_id_table       = std::make_unique<i2e_id_table_type>();
    }
  }

  void priv_construct_persistent_id_tables(mmanager& localm) {
    if constexpr (k_use_eid_table) {
      m_e2i_id_table.reset(localm.construct<e2i_id_table_type>(
          k_e2i_id_table_name)(localm.get_allocator<>()));
      m_local_e2i_id_table.reset(localm.construct<e2i_id_table_type>(
          k_local_e2i_id_table_name)(localm.get_allocator<>()));
      m_i2e_id_table.reset(localm.construct<i2e_id_table_type>(
          k_i2e_id_table_name)(localm.get_allocator<>()));
    }
  }

  void priv_open_persistent_id_tables(mmanager& localm) {
    if constexpr (k_use_eid_table) {
      m_e2i_id_table.reset(
          localm.find<e2i_id_table_type>(k_e2i_id_table_name).first);
      assert(m_e2i_id_table);
      m_local_e2i_id_table.reset(
          localm.find<e2i_id_table_type>(k_local_e2i_id_table_name).first);
      assert(m_local_e2i_id_table);
      m_i2e_id_table.reset(
          localm.find<i2e_id_table_type>(k_i2e_id_table_name).first);
      assert(m_i2e_id_table);
    }
  }

  template <typename value_type>
  static value_type priv_copy_value_to_heap(const value_type& value) {
    value_type copied_value{};
    copied_value = value;
    return copied_value;
  }

  std::function<int(const id_type&)> priv_get_point_partitioner_external()
      const {
    const int size = m_comm.size();
    return [size](const id_type& id) {
      return external_id_partitioner_hasher{}(id) % size;
    };
  }

  internal_point_partitioner priv_get_point_partitioner_internal() const {
    const int size = m_comm.size();
    return [size](const internal_id_type& id) {
      return point_partitioner_internal_id_hasher{}(id) % size;
    };
  }

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

  void priv_add_point_async(const id_type& eid, const point_type& point) {
    if constexpr (k_use_eid_table) {
      const auto owner = priv_get_point_partitioner_external()(eid);
      m_comm.async(
          owner,
          [](auto, auto pthis, const id_type& eid, const point_type& point) {
            const internal_id_type itn_id =
                pthis->priv_find_local_internal_id(eid);
            pthis->priv_add_point_with_internal_id_async(itn_id, point);
          },
          m_this, eid, point);
    } else {
      const auto owner = priv_get_point_partitioner_internal()(eid);
      m_comm.async(
          owner,
          [](auto, auto pthis, const id_type& id, const point_type& point) {
            pthis->priv_add_point_locally(id, point);
          },
          m_this, eid, point);
    }
  }

  void priv_add_point_with_internal_id_async(const internal_id_type& itn_id,
                                             const point_type&       point) {
    static_assert(k_use_eid_table,
                  "priv_add_point_with_internal_id_async() is only available "
                  "when external ID and internal ID are different.");

    const auto owner = priv_get_point_partitioner_internal()(itn_id);
    m_comm.async(
        owner,
        [](auto, auto this_ptr, const internal_id_type id,
           const point_type& sent_point) {
          this_ptr->priv_add_point_locally(id, sent_point);
        },
        m_this, itn_id, point);
  }

  void priv_add_point_locally(const internal_id_type& iid,
                              const point_type&       point) {
    if (m_pstore->contains(iid)) {
      std::cerr << "Duplicate internal ID " << iid << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    (*m_pstore)[iid] = point;
  }

  inline internal_id_type priv_find_local_internal_id(const id_type& id) const {
    if constexpr (!k_use_eid_table) {
      return id;
    } else {
      if (m_local_e2i_id_table->contains(id)) {
        return m_local_e2i_id_table->at(id);
      }
      if (m_e2i_id_table->contains(id)) {
        return m_e2i_id_table->at(id);
      }
      assert(false && "No internal ID mapping for the given external ID");
      return internal_id_type{};
    }
  }

  inline id_type priv_get_external_id_local(const internal_id_type& id) const {
    if constexpr (k_use_eid_table) {
      return priv_copy_value_to_heap(m_i2e_id_table->at(id));
    } else {
      return id;
    }
  }

  template <typename id_iterator>
  boost::unordered::unordered_flat_map<internal_id_type, id_type>
  priv_get_external_ids(id_iterator internal_ids_begin,
                        id_iterator internal_ids_end) const {
    static_assert(
        std::is_same_v<typename std::iterator_traits<id_iterator>::value_type,
                       internal_id_type>,
        "id_iterator must be an iterator of internal_id_type");

    static boost::unordered::unordered_flat_map<internal_id_type, id_type>
        return_id_table;
    return_id_table = decltype(return_id_table){};
    m_comm.cf_barrier();

    for (; internal_ids_begin != internal_ids_end; ++internal_ids_begin) {
      const auto internal_id = *internal_ids_begin;
      const auto owner = priv_get_point_partitioner_internal()(internal_id);
      m_comm.async(
          owner,
          [](auto comm, auto pthis, const internal_id_type internal_id,
             const int source_rank) {
            const auto external_id = pthis->priv_copy_value_to_heap(
                pthis->m_i2e_id_table->at(internal_id));
            comm->async(
                source_rank,
                [](auto, const auto& internal_id, auto external_id) {
                  return_id_table[internal_id] = std::move(external_id);
                },
                internal_id, std::move(external_id));
          },
          m_this, internal_id, m_comm.rank());
    }
    m_comm.barrier();

    return return_id_table;
  }

  template <typename id_iterator>
  boost::unordered::unordered_flat_map<id_type, internal_id_type, hasher>
  priv_get_internal_ids_async(id_iterator external_ids_begin,
                              id_iterator external_ids_end) const {
    static_assert(
        std::is_same_v<typename std::iterator_traits<id_iterator>::value_type,
                       id_type>,
        "id_iterator must be an iterator of id_type");

    static boost::unordered::unordered_flat_map<id_type, internal_id_type,
                                                hasher>
        return_id_table;
    return_id_table = decltype(return_id_table){};
    m_comm.cf_barrier();

    for (; external_ids_begin != external_ids_end; ++external_ids_begin) {
      const auto external_id = *external_ids_begin;
      const auto owner = priv_get_point_partitioner_external()(external_id);
      m_comm.async(
          owner,
          [](auto comm, auto pthis, const id_type external_id,
             const int source_rank) {
            const auto internal_id = pthis->m_e2i_id_table->at(external_id);
            comm->async(
                source_rank,
                [](auto, auto external_id, const internal_id_type internal_id) {
                  return_id_table[std::move(external_id)] = internal_id;
                },
                pthis->priv_copy_value_to_heap(external_id), internal_id);
          },
          m_this, external_id, m_comm.rank());
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

    constexpr internal_id_type k_unassigned_id =
        std::numeric_limits<internal_id_type>::max();

    // First, insert all external IDs into m_e2i_id_table with a dummy internal
    // ID. The internal IDs will be assigned later based on the number of unique
    // external IDs.
    for (auto eit = eids_begin; eit != eids_end; ++eit) {
      const auto eid   = *eit;
      const auto owner = priv_get_point_partitioner_external()(eid);
      m_comm.async(
          owner,
          [](auto, auto pthis, const id_type eid) {
            if (!pthis->m_e2i_id_table->contains(eid)) {
              (*pthis->m_e2i_id_table)[eid] = k_unassigned_id;
            }
          },
          m_this, eid);
    }
    m_comm.barrier();

    // Internal IDs are offset-based ones.
    // Compute offsets first.
    std::size_t num_new_local_ids = 0;
    for (const auto& [eid, int_id] : *m_e2i_id_table) {
      if (int_id == k_unassigned_id) {
        ++num_new_local_ids;
      }
    }
    const auto global_num_new_ids = ygm::sum(num_new_local_ids, m_comm);
    if (global_num_new_ids == 0) {
      return;
    }

    const auto global_num_entries_after =
        ygm::sum(m_e2i_id_table->size(), m_comm);
    const auto global_num_existing_ids =
        global_num_entries_after - global_num_new_ids;
    const auto local_id_offset =
        global_num_existing_ids + ygm::prefix_sum(num_new_local_ids, m_comm);

    // Assign internal IDs to the external IDs.
    std::size_t count = 0;
    for (auto& [eid, int_id] : *m_e2i_id_table) {
      if (int_id != k_unassigned_id) {
        continue;
      }
      int_id = local_id_offset + count;
      ++count;
    }

    // Construct m_i2e_id_table and m_local_e2i_id_table.
    for (const auto& [eid, int_id] : *m_e2i_id_table) {
      if (int_id < global_num_existing_ids) {
        continue;
      }
      const auto owner = priv_get_point_partitioner_internal()(int_id);
      m_comm.async(
          owner,
          [](auto, auto pthis, const internal_id_type int_id,
             const id_type eid) {
            (*pthis->m_i2e_id_table)[int_id]    = eid;
            (*pthis->m_local_e2i_id_table)[eid] = int_id;
          },
          m_this, int_id, eid);
    }
    m_comm.barrier();
  }

  knn_index_type priv_gen_external_knng(
      const internal_knn_index_type& internal_index) const {
    static_assert(
        k_use_eid_table,
        "priv_gen_external_knng() is only available when external ID and "
        "internal ID are different.");

    // Collect all internal IDs that appear in the neighbors of the index
    boost::unordered::unordered_flat_set<internal_id_type> internal_ids;
    for (auto itr = internal_index.points_begin();
         itr != internal_index.points_end(); ++itr) {
      const auto& [internal_id, neighbors] = *itr;
      for (const auto& neighbor : neighbors) {
        internal_ids.insert(neighbor.id);
      }
    }

    knn_index_type external_index;
    const auto     i2e_id_map =
        priv_get_external_ids(internal_ids.begin(), internal_ids.end());
    for (auto itr = internal_index.points_begin();
         itr != internal_index.points_end(); ++itr) {
      const auto& [internal_id, neighbors] = *itr;
      const auto src_eid = priv_get_external_id_local(internal_id);
      for (const auto& neighbor : neighbors) {
        const auto n_eid = priv_copy_value_to_heap(i2e_id_map.at(neighbor.id));
        external_index.insert(
            src_eid, neighbor_type(std::move(n_eid), neighbor.distance));
      }
    }

    return external_index;
  }

  template <typename query_iterator>
  std::pair<neighbor_store_type, internal_neighbor_store_type> priv_run_query(
      const internal_knn_index_type& index,
      distance_function_type distance_function, query_iterator queries_begin,
      query_iterator queries_end, const int k, const double epsilon = 0.1) {
    typename query_kernel_type::option option{.k          = k,
                                              .epsilon    = epsilon,
                                              .mu         = 0,
                                              .batch_size = 1 << 26,
                                              .rnd_seed   = m_rnd_seed,
                                              .verbose    = m_verbose};

    query_kernel_type kernel(option, *m_pstore,
                             priv_get_point_partitioner_internal(),
                             distance_function, index, m_comm);

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

    boost::unordered::unordered_flat_set<internal_id_type> unique_neighbor_ids;
    for (const auto& neighbors : internal_query_result) {
      for (const auto& neighbor : neighbors) {
        unique_neighbor_ids.insert(neighbor.id);
      }
    }

    neighbor_store_type query_result;
    query_result.resize(internal_query_result.size());
    const auto i2e_id_map = priv_get_external_ids(unique_neighbor_ids.begin(),
                                                  unique_neighbor_ids.end());
    for (std::size_t i = 0; i < internal_query_result.size(); ++i) {
      for (const auto& neighbor : internal_query_result[i]) {
        auto eid = priv_copy_value_to_heap(i2e_id_map.at(neighbor.id));
        query_result[i].emplace_back(std::move(eid), neighbor.distance);
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

    static boost::unordered::unordered_flat_map<internal_id_type, point_type>
        neighbor_features;
    neighbor_features = decltype(neighbor_features){};
    m_comm.cf_barrier();
    for (const auto& niid : neighbor_ids) {
      const auto owner = priv_get_point_partitioner_internal()(niid);
      m_comm.async(
          owner,
          [](auto comm, auto pthis, const internal_id_type iid,
             const int source_rank) {
            assert(pthis->m_pstore->contains(iid));

            // Copy into heap-backed storage before sending, which avoids
            // allocator/state issues when the datastore is opened read-only.
            point_type feature;
            feature = pthis->m_pstore->at(iid);

            comm->async(
                source_rank,
                [](auto, const auto& iid, auto feature) {
                  neighbor_features[iid] = std::move(feature);
                },
                iid, std::move(feature));
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

  std::vector<neighbor_type> priv_get_local_external_neighbors(
      const internal_knn_index_type& internal_index,
      const internal_id_type         source_iid) const {
    static_assert(
        k_use_eid_table,
        "priv_get_local_external_neighbors() is only available when external "
        "ID and internal ID are different.");

    boost::unordered::unordered_flat_set<internal_id_type> unique_neighbor_ids;
    for (auto itr = internal_index.neighbors_begin(source_iid),
              end = internal_index.neighbors_end(source_iid);
         itr != end; ++itr) {
      unique_neighbor_ids.insert(itr->id);
    }

    const auto i2e_id_map = priv_get_external_ids(unique_neighbor_ids.begin(),
                                                  unique_neighbor_ids.end());

    std::vector<neighbor_type> neighbors;
    for (auto itr = internal_index.neighbors_begin(source_iid),
              end = internal_index.neighbors_end(source_iid);
         itr != end; ++itr) {
      auto eid = priv_copy_value_to_heap(i2e_id_map.at(itr->id));
      neighbors.emplace_back(std::move(eid), itr->distance);
    }
    return neighbors;
  }

  template <typename id_iterator>
  std::unordered_map<id_type, point_type, external_id_pstore_hasher>
  priv_get_remote_points(id_iterator ids_begin, id_iterator ids_end) const {
    static std::unordered_map<id_type, point_type, external_id_pstore_hasher>
        return_points_store;
    return_points_store = decltype(return_points_store){};
    return_points_store.reserve(std::distance(ids_begin, ids_end));

    auto proc = [](auto comm, auto pthis, const id_type eid,
                   const int source_rank) {
      const auto iid = pthis->priv_find_local_internal_id(eid);
      comm->async(
          pthis->priv_get_point_partitioner_internal()(iid),
          [](auto comm, auto pthis, const internal_id_type iid,
             const id_type eid, const int source_rank) {
            // Copy into heap-backed storage before sending, which avoids
            // allocator/state issues when the datastore is opened read-only.
            point_type point;
            point = pthis->m_pstore->at(iid);

            comm->async(
                source_rank,
                [](auto, auto eid, auto point) {
                  return_points_store[std::move(eid)] = std::move(point);
                },
                pthis->priv_copy_value_to_heap(eid), std::move(point));
          },
          pthis->m_this, iid, pthis->priv_copy_value_to_heap(eid), source_rank);
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

  std::unordered_map<internal_id_type, std::vector<internal_id_type>>
  priv_gen_internal_initial_index(
      const external_initial_index_type& initial_index) const {
    static_assert(
        k_use_eid_table,
        "priv_gen_internal_initial_index() is only available when external "
        "ID and internal ID are different.");

    static std::unordered_map<internal_id_type, std::vector<internal_id_type>>
        internal_initial_index;
    internal_initial_index = decltype(internal_initial_index){};

    boost::unordered::unordered_flat_set<id_type, hasher> unique_external_ids;
    for (const auto& [source_id, neighbors] : initial_index) {
      unique_external_ids.insert(source_id);
      for (const auto& neighbor_id : neighbors) {
        unique_external_ids.insert(neighbor_id);
      }
    }

    const auto e2i_id_map = priv_get_internal_ids_async(
        unique_external_ids.begin(), unique_external_ids.end());
    m_comm.cf_barrier();

    for (const auto& [source_id, neighbors] : initial_index) {
      const auto                    source_iid = e2i_id_map.at(source_id);
      std::vector<internal_id_type> neighbor_iids;
      neighbor_iids.reserve(neighbors.size());
      for (const auto& neighbor_id : neighbors) {
        neighbor_iids.push_back(e2i_id_map.at(neighbor_id));
      }

      m_comm.async(
          priv_get_point_partitioner_internal()(source_iid),
          [](auto, const internal_id_type source_iid,
             const std::vector<internal_id_type>& neighbor_iids) {
            internal_initial_index[source_iid] = neighbor_iids;
          },
          source_iid, neighbor_iids);
    }
    m_comm.barrier();

    return internal_initial_index;
  }

  ygm::comm&                                           m_comm;
  uint64_t                                             m_rnd_seed;
  bool                                                 m_verbose;
  std::unique_ptr<metall::utility::metall_mpi_adaptor> m_metall{nullptr};
  std::unique_ptr<point_store_type>                    m_pstore{nullptr};
  std::unique_ptr<internal_knn_index_container> m_knn_index_list{nullptr};
  std::unique_ptr<size_container>               m_index_k_list{nullptr};
  ygm::ygm_ptr<self_type>                       m_this{this};

  // Use the ID mapping tables only when id_type is not an integral type.
  //
  // Contains the mapping from external ID to internal ID.
  // Note: Owning an external ID does not necessarily mean owning the
  // corresponding point data. The owner of a point data is determined by the
  // point partitioner, which is based on the internal ID.
  std::unique_ptr<e2i_id_table_type> m_e2i_id_table{nullptr};
  // Mapping from internal ID to external ID. The associated points are stored
  // locally.
  std::unique_ptr<i2e_id_table_type> m_i2e_id_table{nullptr};
  // Mapping from external ID to internal ID.  The associated points are stored
  // locally.
  std::unique_ptr<e2i_id_table_type> m_local_e2i_id_table{nullptr};
};

}  // namespace saltatlas
