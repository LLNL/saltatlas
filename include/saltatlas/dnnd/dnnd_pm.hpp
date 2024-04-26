// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

// DNND does not use multi-threading within an MPI rank.
// Thus, we disable concurrency in Metall to achieve better performance.
#define METALL_DISABLE_CONCURRENCY

#include <memory>
#include <random>
#include <string_view>

#include <ygm/comm.hpp>

#include <metall/utility/metall_mpi_adaptor.hpp>

#include <saltatlas/dnnd/detail/base_dnnd.hpp>

namespace saltatlas {

/// \brief Persistent Distributed NNDescent.
/// \tparam Id Point ID type.
/// \tparam PointType Point type.
/// \tparam Distance Distance type.
template <typename Id = uint64_t, typename PointType = feature_vector<double>,
          typename Distance = double>
class dnnd_pm : public dndetail::base_dnnd<
                    Id, PointType, Distance,
                    metall::manager::fallback_allocator<std::byte>> {
 private:
  using base_type =
      dndetail::base_dnnd<Id, PointType, Distance,
                          metall::manager::fallback_allocator<std::byte>>;
  using data_core_type = typename base_type::data_core_type;

 public:
  using id_type             = typename base_type::id_type;
  using point_type          = typename base_type::point_type;
  using distance_type       = typename base_type::distance_type;
  using point_store_type    = typename base_type::point_store_type;
  using knn_index_type      = typename base_type::knn_index_type;
  using neighbor_type       = typename base_type::neighbor_type;
  using query_store_type    = typename base_type::query_store_type;
  using point_partitioner   = typename base_type::point_partitioner;
  using neighbor_store_type = typename base_type::neighbor_store_type;

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

  /// \brief Constructor. Create a new persistent index.
  /// \param datastore_path Path to store data.
  /// It is not allowed to contain another index at the path.
  /// \param distance_name Distance metric name.
  /// \param comm YGM comm instance.
  /// \param rnd_seed Seed for random generators.
  /// \param verbose If true, enable the verbose mode.
  dnnd_pm(create_t, const std::string_view datastore_path,
          const std::string_view distance_name, ygm::comm& comm,
          const uint64_t rnd_seed = std::random_device{}(),
          const bool     verbose  = false)
      : base_type(verbose, comm) {
    priv_create(datastore_path, distance_name, rnd_seed);
    base_type::init_data_core(*m_data_core);
    comm.cf_barrier();
  }

  /// \brief Constructor. Opens an already stored index.
  /// \param datastore_path Path to existing index.
  /// \param comm YGM comm instance.
  /// \param verbose If true, enable the verbose mode.
  dnnd_pm(open_t, const std::string_view datastore_path, ygm::comm& comm,
          const bool verbose = false)
      : base_type(verbose, comm) {
    priv_open(datastore_path);
    base_type::init_data_core(*m_data_core);
    comm.cf_barrier();
  }

  /// \brief Constructor. Opens an already stored index with the read-only mode.
  /// \param datastore_path Path to existing index.
  /// \param comm YGM comm instance.
  /// \param verbose If true, enable the verbose mode.
  dnnd_pm(open_read_only_t, const std::string_view datastore_path,
          ygm::comm& comm, const bool verbose = false)
      : base_type(verbose, comm) {
    priv_open_read_only(datastore_path);
    base_type::init_data_core(*m_data_core);
    comm.cf_barrier();
  }

  /// \brief Takes a snapshot.
  /// \param destination_path Destination path.
  /// \return Returns true on success; otherwise, false.
  bool snapshot(const std::string_view& destination_path) {
    return m_metall->snapshot(destination_path.data());
  }

  /// \brief Destroy the dataset from the datastore.
  void destroy_dataset() {
    std::destroy_at(&(m_data_core->pstore));
    base_type::get_comm().cf_barrier();
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
                   const std::string_view distance_name,
                   const uint64_t         rnd_seed) {
    priv_create_metall(path);
    auto& lmgr  = m_metall->get_local_manager();
    m_data_core = lmgr.construct<data_core_type>(metall::unique_instance)(
        distance::convert_to_distance_id(distance_name), rnd_seed,
        lmgr.get_allocator());
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

  std::unique_ptr<metall::utility::metall_mpi_adaptor> m_metall{nullptr};
  data_core_type*                                      m_data_core{nullptr};
};

}  // namespace saltatlas
