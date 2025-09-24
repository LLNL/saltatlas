// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <boost/unordered/unordered_flat_map.hpp>
#include <filesystem>
#include <memory>
#include <metall/metall.hpp>
#include <vector>

#include "../mpi.hpp"
#include "saltatlas/common/detail/utilities/hash.hpp"
#include "saltatlas/neo_dnnd/detail/utilities/shm_manager.hpp"

namespace saltatlas::dndetail {

/// \brief Interprocess shared feature vector pool.
/// There are multiple banks, each bank can contain the same number of feature
/// vectors.
template <typename id_type, typename feature_element_type>
class pfv_replica_store {
 private:
  using index_type = uint32_t;
  using seg_allocator = metall::manager::allocator_type<std::byte>;
  using index_table_type = boost::unordered::unordered_flat_map<
      id_type, index_type, saltatlas::hash<1314>, std::equal_to<>,
      typename std::allocator_traits<seg_allocator>::template rebind_alloc<
          std::pair<const id_type, index_type>>>;

 public:
  /// \param dims Dimension size of feature vectors.
  /// \param single_capacity Maximum number of feature vectors per bank.
  /// \param num_banks Number of banks.
  /// \param my_bank_no My bank number [0, num_banks).
  /// \param cache_name Name of the shared memory region.
  /// \param comm MPI communicator for synchronization.
  pfv_replica_store(const size_t dims, const size_t single_capacity,
                    const size_t num_banks, const size_t my_bank_no,
                    const std::string& cache_name, mpi::communicator& comm)
      : m_dims(dims),
        m_single_capacity(single_capacity),
        m_num_banks(num_banks),
        m_my_bank_no(my_bank_no),
        m_cache_name(cache_name),
        m_comm(comm) {
    // check index type size
    if (single_capacity > std::numeric_limits<index_type>::max()) {
      std::cerr << "Too large capacity per bank: " << single_capacity
                << std::endl;
      std::abort();
    }
    if (my_bank_no >= num_banks) {
      std::cerr << "Invalid bank number: " << my_bank_no << std::endl;
      std::abort();
    }
    m_segment_managers.resize(m_num_banks, nullptr);
    m_fv_pools.resize(m_num_banks, nullptr);
    m_index_tables.resize(m_num_banks, nullptr);
  }

  ~pfv_replica_store() {
    m_fv_pools.clear();
    m_index_tables.clear();
    for (auto* manager : m_segment_managers) {
      delete manager;
    }
    m_comm.node_local_comm();
  }

  /// \brief Create my bank with write access.
  void create_mine() {
    m_segment_managers[m_my_bank_no] = new metall::manager(
        metall::create_only, priv_gen_shm_name(m_my_bank_no));
    auto* manager = m_segment_managers[m_my_bank_no];

    m_fv_pools[m_my_bank_no] =
        static_cast<feature_element_type*>(manager->allocate(
            m_dims * m_single_capacity * sizeof(feature_element_type)));
    manager->construct<metall::offset_ptr<feature_element_type>>(
        metall::unique_instance)(m_fv_pools[m_my_bank_no]);

    m_index_tables[m_my_bank_no] = manager->construct<index_table_type>(
        metall::unique_instance)(manager->template get_allocator<std::byte>());
    m_index_tables[m_my_bank_no]->reserve(m_single_capacity);
  }

  /// \brief Close my bank (after create_mine()).
  void close_mine() {
    delete m_segment_managers[m_my_bank_no];
    m_segment_managers[m_my_bank_no] = nullptr;
    m_comm.node_local_comm();
  }

  /// \brief Open all banks with read-only access.
  /// This function should be called after close_mine().
  void open_all_read_only() {
    for (int i = 0; i < m_num_banks; ++i) {
      m_segment_managers[i] =
          new metall::manager(metall::open_read_only, priv_gen_shm_name(i));
      auto* pool_offset_ptr =
          m_segment_managers[i]
              ->find<metall::offset_ptr<feature_element_type>>(
                  metall::unique_instance)
              .first;
      m_fv_pools[i] = metall::to_raw_pointer(*pool_offset_ptr);
      assert(m_fv_pools[i]);
      m_index_tables[i] = m_segment_managers[i]
                              ->find<index_table_type>(metall::unique_instance)
                              .first;
      assert(m_index_tables[i]);
    }
    m_comm.node_local_comm();
  }

  /// \brief Register a feature vector ID to my bank.
  /// \param id Feature vector ID.
  /// \return true if the ID is newly registered, false if the ID already exists
  bool register_id(const id_type id) {
    const auto bank_no = m_my_bank_no;
    if (m_index_tables.at(bank_no)->count(id) > 0) {
      return false;
    }
    const auto index = m_index_tables.at(bank_no)->size();
    (*(m_index_tables.at(bank_no)))[id] = index;
    return true;
  }

  /// \brief Get a feature vector by ID from a specified bank.
  /// \param bank_no Bank number [0, num_banks).
  /// \param id Feature vector ID.
  /// \return Pointer to the feature vector.
  /// \note This function assumes that the ID exists in the specified bank.
  const feature_element_type* get(const int bank_no, const id_type id) const {
    assert(bank_no < m_num_banks);
    assert(bank_no < m_index_tables.size());
    assert(m_index_tables.at(bank_no)->count(id) > 0);
    const auto idx = m_index_tables.at(bank_no)->at(id);
    return m_fv_pools.at(bank_no) + idx * m_dims;
  }

  /// \brief Get my feature vector pool.
  /// \return Pointer to the feature vector pool.
  /// \note This function assumes that create_mine() has been called.
  feature_element_type* my_fv_pool() const {
    return m_fv_pools.at(m_my_bank_no);
  }

  /// \brief Get the number of feature vectors in a specified bank.
  /// \param bank_no Bank number [0, num_banks).
  std::size_t size(const int bank_no) const {
    return m_index_tables.at(bank_no)->size();
  }

 private:
  std::string priv_gen_shm_name(const int id) const {
    std::filesystem::path path;
#ifdef __APPLE__
    path = "/tmp/";
#else
    path = "/dev/shm/";
#endif
    path /= "neo_dnnd-pcache" + m_cache_name + "-" + std::to_string(id);
    return path.string();
  }

  const size_t m_dims;
  const size_t m_single_capacity;
  const size_t m_num_banks;
  const size_t m_my_bank_no;
  const std::string m_cache_name;
  mpi::communicator& m_comm;
  std::vector<feature_element_type*> m_fv_pools;
  std::vector<index_table_type*> m_index_tables;
  std::vector<metall::manager*> m_segment_managers;
};

}  // namespace saltatlas::dndetail