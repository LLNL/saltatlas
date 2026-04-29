// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "saltatlas/common/detail/utilities/general.hpp"

#define DNND2_CHECK_MPI(ret)                                                  \
  do {                                                                        \
    if (ret != MPI_SUCCESS) {                                                 \
      std::cerr << __FILE__ << ":" << __LINE__ << " MPI error." << std::endl; \
      ::MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                              \
    }                                                                         \
  } while (0)

namespace saltatlas::mpi {

/// \brief Returns the MPI_Datatype for the given type.
struct data_type {
  template <typename T>
  static constexpr MPI_Datatype get() {
    if constexpr (std::is_same_v<T, std::byte>) {
      return MPI_BYTE;
    } else if constexpr (std::is_same_v<T, bool>) {
      return MPI_C_BOOL;
    } else if constexpr (std::is_same_v<T, char>) {
      return MPI_CHAR;
    } else if constexpr (std::is_same_v<T, unsigned char>) {
      return MPI_UNSIGNED_CHAR;
    } else if constexpr (std::is_same_v<T, int>) {
      return MPI_INT;
    } else if constexpr (std::is_same_v<T, unsigned int>) {
      return MPI_UNSIGNED;
    } else if constexpr (std::is_same_v<T, long>) {
      return MPI_LONG;
    } else if constexpr (std::is_same_v<T, unsigned long>) {
      return MPI_UNSIGNED_LONG;
    } else if constexpr (std::is_same_v<T, long long> ||
                         std::is_same_v<T, long long int> ||
                         std::is_same_v<T, long long unsigned int> ||
                         std::is_same_v<T, std::size_t>) {
      return MPI_LONG_LONG_INT;
    } else if constexpr (std::is_same_v<T, float>) {
      return MPI_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
      return MPI_DOUBLE;
    } else if constexpr (std::is_same_v<T, long double>) {
      return MPI_LONG_DOUBLE;
    }
    return MPI_DATATYPE_NULL;
  }
};

void free_mpi_comm(MPI_Comm& comm) { DNND2_CHECK_MPI(::MPI_Comm_free(&comm)); }

void free_mpi_datatype(MPI_Datatype& type) {
  DNND2_CHECK_MPI(::MPI_Type_free(&type));
}

class communicator {
 public:
  explicit communicator(MPI_Comm comm = MPI_COMM_WORLD) : m_comm(comm) {
    DNND2_CHECK_MPI(::MPI_Comm_rank(m_comm, &m_rank));
    DNND2_CHECK_MPI(::MPI_Comm_size(m_comm, &m_size));
    priv_get_node_local_comm_info();
  }

  communicator(const communicator&)            = delete;
  communicator& operator=(const communicator&) = delete;
  communicator(communicator&&)                 = delete;
  communicator& operator=(communicator&&)      = delete;

  ~communicator() {
    barrier();
    free_mpi_comm(m_node_local_comm);
    barrier();
  }

  int rank() const { return m_rank; }
  int size() const { return m_size; }

  int node_local_rank() const { return m_node_local_rank; }
  int node_size() const { return m_node_local_size; }

  int num_nodes() const {
    assert(m_size % node_size() == 0);
    return m_size / node_size();
  }

  int node_rank() const { return rank() / node_size(); }

  /// \brief Returns std::cout if the rank is 0.
  /// \return std::cout if the rank is 0, otherwise std::nullopt.
  std::ostream& cout0() const {
    static std::ostringstream dummy;
    if (m_rank == 0) {
      return std::cout;
    } else {
      return dummy;
    }
  }

  std::ostream& cerr0() const {
    static std::ostringstream dummy;
    if (m_rank == 0) {
      return std::cerr;
    } else {
      return dummy;
    }
  }

  std::ostream& cout() const {
    std::cout << "[" << m_rank << "]: ";
    return std::cout;
  }

  std::ostream& cerr() const {
    std::cout << "[" << m_rank << "]: ";
    return std::cout;
  }

  void barrier() const { DNND2_CHECK_MPI(::MPI_Barrier(m_comm)); }

  void node_local_barrier() const {
    DNND2_CHECK_MPI(::MPI_Barrier(m_node_local_comm));
  }

  void abort() const { DNND2_CHECK_MPI(::MPI_Abort(m_comm, EXIT_FAILURE)); }

  MPI_Comm comm() const { return m_comm; }

  MPI_Comm node_local_comm() const { return m_node_local_comm; }

  template <typename T>
  void isend(const std::vector<T>& buf, const int dest,
             MPI_Request& request) const {
    DNND2_CHECK_MPI(::MPI_Isend(buf.data(), buf.size(), data_type::get<T>(),
                                dest, 0, m_comm, &request));
  }

  template <typename T>
  void isend(const T* const buf, const int count, const int dest,
             MPI_Request& request) const {
    DNND2_CHECK_MPI(::MPI_Isend(buf, count, data_type::get<T>(), dest, 0,
                                m_comm, &request));
  }

  template <typename T>
  void isend(const T* const buf, const int count, ::MPI_Datatype& type,
             const int dest, MPI_Request& request) const {
    DNND2_CHECK_MPI(::MPI_Isend(buf, count, type, dest, 0, m_comm, &request));
  }

  template <typename T>
  void irecv(const int from, T* const buf, const int count,
             MPI_Request& request) const {
    DNND2_CHECK_MPI(::MPI_Irecv(buf, count, data_type::get<T>(), from, 0,
                                m_comm, &request));
  }

  template <typename T>
  void irecv(const int from, std::vector<T>& buf, MPI_Request& request) const {
    const auto count = priv_get_count(from, data_type::get<T>());
    buf.resize(count);
    DNND2_CHECK_MPI(::MPI_Irecv(buf.data(), count, data_type::get<T>(), from, 0,
                                m_comm, &request));
  }
  template <typename T>
  void irecv(const int from, std::vector<T>& buf, ::MPI_Datatype& type,
             MPI_Request& request) const {
    const auto count = priv_get_count(from, type);
    buf.resize(count);
    DNND2_CHECK_MPI(
        ::MPI_Irecv(buf.data(), count, type, from, 0, m_comm, &request));
  }

  void wait(MPI_Request& request) const {
    DNND2_CHECK_MPI(::MPI_Wait(&request, MPI_STATUS_IGNORE));
  }

  template <typename T>
  void all_reduce(const T& send, T& recv, MPI_Op op) const {
    DNND2_CHECK_MPI(
        ::MPI_Allreduce(&send, &recv, 1, data_type::get<T>(), op, m_comm));
  }

  template <typename T>
  void all_reduce(const T* const send_buf, T* const recv_buf, const int count,
                  MPI_Op op) const {
    DNND2_CHECK_MPI(::MPI_Allreduce(send_buf, recv_buf, count,
                                    data_type::get<T>(), op, m_comm));
  }

  template <typename T>
  std::vector<T> all_reduce(const std::vector<T>& send_buf, MPI_Op op) const {
    std::vector<T> recv_buf;
    recv_buf.resize(send_buf.size());
    all_reduce(send_buf.data(), recv_buf.data(), send_buf.size(), op);
    return recv_buf;
  }

  template <typename T>
  T all_reduce_sum(const T send) const {
    T sum;
    all_reduce(send, sum, MPI_SUM);
    return sum;
  }

  template <typename T>
  T all_reduce_min(const T send) const {
    T min;
    all_reduce(send, min, MPI_MIN);
    return min;
  }

  template <typename T>
  T all_reduce_max(const T send) const {
    T max;
    all_reduce(send, max, MPI_MAX);
    return max;
  }

  // Only the node local root ranks send the data.
  template <typename T>
  T all_node_reduce_sum(const T send) const {
    T       sum;
    const T buf = node_local_rank() == 0 ? send : 0;
    all_reduce(buf, sum, MPI_SUM);
    return sum;
  }

  // Only the node local root ranks send the data.
  template <typename T>
  T all_node_reduce_max(const T send) const {
    T       max;
    const T buf = node_local_rank() == 0 ? send : std::numeric_limits<T>::min();
    all_reduce(buf, max, MPI_MAX);
    return max;
  }

  // Only the node local root ranks send the data.
  template <typename T>
  T all_node_reduce_min(const T send) const {
    T       min;
    const T buf = node_local_rank() == 0 ? send : std::numeric_limits<T>::max();
    all_reduce(buf, min, MPI_MIN);
    return min;
  }

  template <typename T>
  T reduce_sum(const T& send, const int root = 0) const {
    T recv;
    DNND2_CHECK_MPI(::MPI_Reduce(&send, &recv, 1, data_type::get<T>(), MPI_SUM,
                                 root, m_comm));
    return recv;
  }

  template <typename T>
  std::vector<T> reduce_sum(const std::vector<T>& send, const int root) const {
    std::vector<T> recv;
    recv.resize(send.size());
    DNND2_CHECK_MPI(::MPI_Reduce(send.data(), recv.data(), send.size(),
                                 data_type::get<T>(), MPI_SUM, root, m_comm));
    return recv;
  }

  template <class T>
  std::vector<T> gather(const T& send, const int root = 0) const {
    std::vector<T> recv_buf;
    if (m_rank == root) {
      recv_buf.resize(size());
      gather(send, recv_buf.data(), root);
    } else {
      gather(send, static_cast<T*>(nullptr), root);
    }
    return recv_buf;
  }

  template <class T>
  void gather(const T& send, T* const recv_buf, const int root = 0) const {
    MPI_Comm cm = comm();
    DNND2_CHECK_MPI(::MPI_Gather(&send, 1, data_type::get<T>(), recv_buf, 1,
                                 data_type::get<T>(), root, cm));
  }

  template <class T>
  void all_gather(const T& send, std::vector<T>& recv_buf,
                  const bool global = true) const {
    recv_buf.resize(global ? size() : node_size());
    all_gather(send, recv_buf.data(), global);
  }

  template <class T>
  void all_gather(const T& send, T* const recv_buf,
                  const bool global = true) const {
    MPI_Comm cm = (global) ? comm() : node_local_comm();
    DNND2_CHECK_MPI(::MPI_Allgather(&send, 1, data_type::get<T>(), recv_buf, 1,
                                    data_type::get<T>(), cm));
  }

  template <class T>
  void all_gather_v(const T* const send_buf, const std::size_t send_count,
                    std::vector<T>& recv_buf, const bool global = true) const {
    MPI_Comm cm = (global) ? comm() : node_local_comm();

    if (send_count >= std::numeric_limits<int>::max()) {
      std::cerr << __FILE__ << " : " << __LINE__
                << " Too large data size to send: " << send_count << std::endl;
      std::abort();
    }

    const auto       sz = global ? size() : node_size();
    std::vector<int> recv_counts(sz);
    all_gather(int(send_count), recv_counts.data(), cm);

    std::vector<int> displs(sz);
    displs[0] = 0;
    for (int i = 1; i < sz; ++i) {
      displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
    const auto recv_count = displs.back() + recv_counts.back();
    recv_buf.resize(recv_count);

    DNND2_CHECK_MPI(::MPI_Allgatherv(send_buf, send_count, data_type::get<T>(),
                                     recv_buf.data(), recv_counts.data(),
                                     displs.data(), data_type::get<T>(), cm));
  }

  template <class T>
  void bcast(T& data, const int root) const {
    DNND2_CHECK_MPI(::MPI_Bcast(&data, 1, data_type::get<T>(), root, m_comm));
  }

  // bcast, void* version
  void bcast_bytes(void* const data, const int count, const int root) const {
    DNND2_CHECK_MPI(::MPI_Bcast(data, count, MPI_BYTE, root, m_comm));
  }

  /// \brief Send and receive data of arbitrary size to and from a pair rank.
  /// If the size of the data is larger than 'batch_size_byte',
  /// the data is sent in batches.
  /// TODO: reserve buffer to avoid reallocations.
  template <typename T>
  void sendrecv_arb_size(const int pair_rank, const std::vector<T>& send_buffer,
                         std::vector<T>&   recv_buffer,
                         ::MPI_Datatype    data_type = data_type::get<T>(),
                         const std::size_t batch_size_byte = 1 << 26) {
    if (pair_rank == m_rank) {
      recv_buffer = send_buffer;
      return;
    }

    const std::size_t batch_size = batch_size_byte / sizeof(T);
    const int num_batches  = (send_buffer.size() + batch_size - 1) / batch_size;
    bool      received_all = false;
    recv_buffer.clear();
    for (int i = 0; i < num_batches || !received_all; ++i) {
      const bool reached_last = i >= num_batches - 1;
      const auto off          = std::min(batch_size * i, send_buffer.size());
      const auto send_size    = std::min(batch_size, send_buffer.size() - off);
      received_all |= priv_sendrecv_arb_size_helper(
          pair_rank, send_buffer.data() + off, send_size, reached_last,
          data_type, recv_buffer);
    }
  }

  /// \brief sendrecv_arb_size optimized version (use move on the same rank).
  template <typename T>
  void sendrecv_arb_size_opt(const int pair_rank, std::vector<T>&& send_buffer,
                             std::vector<T>&   recv_buffer,
                             ::MPI_Datatype    data_type = data_type::get<T>(),
                             const std::size_t batch_size_byte = 1 << 26) {
    if (pair_rank == m_rank) {
      recv_buffer = std::move(send_buffer);
      return;
    }
    sendrecv_arb_size(pair_rank, send_buffer, recv_buffer, data_type,
                      batch_size_byte);
  }

  // Every rank sends the same number of elements to every other rank.
  template <typename T>
  void all_to_all(const std::vector<T>& send_buffer,
                  std::vector<T>&       recv_buffer) {
    if (send_buffer.size() * sizeof(T) > std::numeric_limits<int>::max()) {
      std::cerr << "Too large data size to send: "
                << send_buffer.size() * sizeof(T) << std::endl;
      abort();
    }

    const size_t count = send_buffer.size() / m_size;
    recv_buffer.resize(send_buffer.size());
    DNND2_CHECK_MPI(::MPI_Alltoall(send_buffer.data(), count,
                                   data_type::get<T>(), recv_buffer.data(),
                                   count, data_type::get<T>(), m_comm));
  }

  // Return the number of elements received from each rank
  template <typename T>
  std::vector<int> all_to_all_v(const std::vector<T>&   send_buffer,
                                const std::vector<int>& send_counts,
                                std::vector<T>&         recv_buffer) {
    if (send_buffer.size() * sizeof(T) > (1ULL << 31ULL)) {
      std::cerr << "Too large data size to send: "
                << send_buffer.size() * sizeof(T) << std::endl;
      abort();
    }
    assert(send_counts.size() == (std::size_t)m_size);

    auto sdispls = send_counts;
    std::partial_sum(sdispls.cbegin(), sdispls.cend(), sdispls.begin());
    sdispls.insert(sdispls.begin(), 0);

    std::vector<int> recv_counts;
    all_to_all(send_counts, recv_counts);

    auto rdispls = recv_counts;
    std::partial_sum(rdispls.cbegin(), rdispls.cend(), rdispls.begin());
    rdispls.insert(rdispls.begin(), 0);
    recv_buffer.resize(rdispls.back());

    DNND2_CHECK_MPI(::MPI_Alltoallv(
        send_buffer.data(), send_counts.data(), sdispls.data(),
        data_type::get<T>(), recv_buffer.data(), recv_counts.data(),
        rdispls.data(), data_type::get<T>(), m_comm));

    return recv_counts;
  }

  // Return the number of elements received from each rank
  // This function copies the data to send_buffer. So, it could be slow.
  template <typename T>
  std::vector<int> all_to_all_v(const std::vector<std::vector<T>>& to_send,
                                std::vector<T>& recv_buffer) {
    std::vector<int> send_counts(m_size, 0);
    std::size_t      num_sends = 0;
    for (int r = 0; r < to_send.size(); ++r) {
      send_counts[r] = to_send[r].size();
      num_sends += to_send[r].size();
    }

    std::vector<T> send_buf(num_sends);
    size_t         off = 0;
    for (int r = 0; r < to_send.size(); ++r) {
      std::copy(to_send[r].begin(), to_send[r].end(), send_buf.begin() + off);
      off += to_send[r].size();
    }
    return all_to_all_v(send_buf, send_counts, recv_buffer);
  }

  void show_mpi_info() const {
    cout0() << "MPI Info" << std::endl;
    cout0() << "  #of ranks: " << size() << std::endl;
    cout0() << "  #of nodes: " << num_nodes() << std::endl;
    cout0() << "  Node size: " << node_size() << std::endl;

    cout0() << "  Rank\tLocal rank" << std::endl;
    for (int i = 0; i < size(); ++i) {
      barrier();
      if (i == rank()) {
        std::cout << "  \t" << rank() << "\t" << node_local_rank() << std::endl;
      }
    }
    barrier();
  }

 private:
  void priv_get_node_local_comm_info() {
    DNND2_CHECK_MPI(::MPI_Comm_split_type(m_comm, MPI_COMM_TYPE_SHARED, m_rank,
                                          MPI_INFO_NULL, &m_node_local_comm));
    DNND2_CHECK_MPI(::MPI_Comm_size(m_node_local_comm, &m_node_local_size));
    DNND2_CHECK_MPI(::MPI_Comm_rank(m_node_local_comm, &m_node_local_rank));
  }

  int priv_get_count(const int from, ::MPI_Datatype type) const {
    MPI_Status status;
    DNND2_CHECK_MPI(::MPI_Probe(from, MPI_ANY_TAG, m_comm, &status));
    int count;
    DNND2_CHECK_MPI(::MPI_Get_count(&status, type, &count));
    return count;
  }

  template <typename T>
  int priv_sendrecv_arb_size_helper(const int       pair_rank,
                                    const T* const  send_buffer,
                                    const int       send_count,
                                    const bool      reached_last,
                                    ::MPI_Datatype  data_type,
                                    std::vector<T>& recv_buffer) {
    MPI_Request isend_request;
    if (send_count * sizeof(T) > (1ULL << 31ULL)) {
      std::cerr << "Too large data size to send: " << send_count * sizeof(T)
                << std::endl;
      abort();
    }
    DNND2_CHECK_MPI(::MPI_Isend(send_buffer, send_count, data_type, pair_rank,
                                reached_last ? 1 : 0, m_comm, &isend_request));

    MPI_Status status;
    DNND2_CHECK_MPI(::MPI_Probe(pair_rank, MPI_ANY_TAG, m_comm, &status));
    int count;
    DNND2_CHECK_MPI(::MPI_Get_count(&status, data_type, &count));
    assert(count >= 0);

    const auto off = recv_buffer.size();
    recv_buffer.resize(count + off);

    DNND2_CHECK_MPI(::MPI_Recv(recv_buffer.data() + off, count, data_type,
                               pair_rank, MPI_ANY_TAG, m_comm,
                               MPI_STATUS_IGNORE));
    DNND2_CHECK_MPI(::MPI_Wait(&isend_request, MPI_STATUS_IGNORE));

    const bool received_last = (status.MPI_TAG == 1);

    return received_last;
  }

  ::MPI_Comm m_comm;
  ::MPI_Comm m_node_local_comm;
  int        m_rank;
  int        m_size;
  int        m_node_local_rank;
  int        m_node_local_size;
};

/// \brief Execute a user-defined function for each unique pair of ranks using
/// the round-robin tournament algorithm.
// The user-defined function is executed as 'size' times (including
// self-directed communication). In every execution, each rank is exclusively
// paired with one other rank. If rank ‘a' is paired with rank ‘b', rank ‘b' is
// paired with only rank ‘a' during the same step. Thus, all ranks can execute
// the function, utilizing the parallelism fully.
/// \tparam function_t Function type.
/// \param comm_size MPI size. Must be 1 or an even number.
/// \param comm_rank My MPI rank.
/// \param func Function to execute. Takes a rank as an argument.
template <typename function_t>
inline void pair_wise_all_to_all(const int comm_size, const int comm_rank,
                                 const function_t& func,
                                 const MPI_Comm    comm      = MPI_COMM_WORLD,
                                 const bool        skip_self = false) {
  if (!skip_self) {
    // self-directed communication
    func(comm_rank);
  }
  if (comm_size == 1) return;

  if (comm_size % 2 != 0) {
    std::cerr << "MPI size must be even" << std::endl;
    DNND2_CHECK_MPI(::MPI_Abort(comm, EXIT_FAILURE));
  }

  std::vector<int> pairs(comm_size - 1, -1);
  saltatlas::detail::gen_round_robin_tournament(comm_size, comm_rank,
                                                pairs.begin());
  for (const auto pair_rank : pairs) {
    func(pair_rank);
  }
}

inline std::vector<int> get_pair_wise_all_to_all_pattern(const int comm_size,
                                                         const int comm_rank) {
  std::vector<int> table;
  pair_wise_all_to_all(comm_size, comm_rank, [&](const int pair_rank) {
    table.push_back(pair_rank);
  });
  return table;
}

inline void show_task_distribution(const std::vector<std::size_t>& table) {
  const auto sum  = std::accumulate(table.begin(), table.end(), std::size_t(0));
  const auto mean = (double)sum / (double)table.size();
  std::cout << "Assigned " << sum << " tasks to " << table.size() << " workers"
            << std::endl;
  std::cout << "Max, Mean, Min:\t"
            << "" << *std::max_element(table.begin(), table.end()) << ", "
            << mean << ", " << *std::min_element(table.begin(), table.end())
            << std::endl;
  double x = 0;
  for (const auto n : table) x += std::pow(n - mean, 2);
  const auto dv = std::sqrt(x / table.size());
  std::cout << "Standard Deviation " << dv << std::endl;
}

/// \brief Compute the number of tasks each MPI rank works on.
/// \param num_local_tasks #of tasks in local.
/// \param batch_size Global batch size. Up to this number of tasks are
/// assigned over all ranks. If 0 is specified, all tasks are assigned. \param
/// mpi_rank My MPI rank. \param mpi_size MPI size. \param verbose Verbose
/// mode. \return #of tasks assigned to myself.
inline std::size_t assign_tasks(const std::size_t num_local_tasks,
                                const std::size_t batch_size,
                                const int mpi_rank, const int mpi_size,
                                const bool     verbose,
                                const MPI_Comm mpi_comm = MPI_COMM_WORLD) {
  if (batch_size == 0) {
    return num_local_tasks;
  }

  std::size_t local_num_assigned_tasks = 0;
  if (mpi_rank > 0) {
    // Send the number of tasks to process to rank 0.
    DNND2_CHECK_MPI(
        ::MPI_Send(&num_local_tasks, 1, MPI_UNSIGNED_LONG, 0, 0, mpi_comm));

    // Receive the number of assigned tasks to process from rank 0.
    MPI_Status status;
    DNND2_CHECK_MPI(::MPI_Recv(&local_num_assigned_tasks, 1, MPI_UNSIGNED_LONG,
                               0, 0, mpi_comm, &status));
  } else {
    // Gather the number of tasks each MPI has
    std::vector<std::size_t> num_remaining_tasks_table(mpi_size, 0);
    num_remaining_tasks_table[0] = num_local_tasks;
    for (int r = 1; r < mpi_size; ++r) {
      MPI_Status status;
      DNND2_CHECK_MPI(::MPI_Recv(&num_remaining_tasks_table[r], 1,
                                 MPI_UNSIGNED_LONG, r, 0, mpi_comm, &status));
    }

    const auto num_global_tasks =
        std::accumulate(num_remaining_tasks_table.begin(),
                        num_remaining_tasks_table.end(), std::size_t(0));
    assert(batch_size > 0);
    std::size_t num_global_unassigned_tasks =
        std::min(batch_size, num_global_tasks);

    // Assigned tasks
    std::vector<std::size_t> task_assignment_table(mpi_size, 0);
    while (num_global_unassigned_tasks > 0) {
      const std::size_t max_num_tasks_per_rank =
          (num_global_unassigned_tasks < (std::size_t)mpi_size)
              ? 1
              : (num_global_unassigned_tasks + mpi_size - 1) / mpi_size;
      for (std::size_t r = 0; r < num_remaining_tasks_table.size(); ++r) {
        const auto n =
            std::min({max_num_tasks_per_rank, num_remaining_tasks_table[r],
                      num_global_unassigned_tasks});
        num_remaining_tasks_table[r] -= n;
        task_assignment_table[r] += n;
        num_global_unassigned_tasks -= n;
      }
    }

    // Tell the computed numbers to the other ranks
    for (int r = 1; r < mpi_size; ++r) {
      DNND2_CHECK_MPI(::MPI_Send(&task_assignment_table[r], 1,
                                 MPI_UNSIGNED_LONG, r, 0, mpi_comm));
    }
    local_num_assigned_tasks = task_assignment_table[0];

    if (verbose) {
      const auto n =
          std::accumulate(task_assignment_table.begin(),
                          task_assignment_table.end(), (std::size_t)0);
      std::cout << "#of total task\t" << num_global_tasks << std::endl;
      std::cout << "#of total assigned task\t" << n << std::endl;
      std::cout << "#of unassigned tasks\t" << num_global_tasks - n
                << std::endl;
      show_task_distribution(task_assignment_table);
    }
  }

  assert(local_num_assigned_tasks <= num_local_tasks);

  DNND2_CHECK_MPI(::MPI_Barrier(mpi_comm));

  return local_num_assigned_tasks;
}

// void win_fence(MPI_Win& win) { DNND2_CHECK_MPI(::MPI_Win_fence(0, win)); }

MPI_Win create_mpi_win(void* base, const std::size_t count, const int disp_unit,
                       MPI_Info info = MPI_INFO_NULL,
                       MPI_Comm comm = MPI_COMM_WORLD) {
  assert(disp_unit > 0);
  if (count * disp_unit > std::numeric_limits<int>::max()) {
    std::cerr << __FILE__ << ":" << __LINE__ << " too large to handle with int"
              << std::endl;
    ::MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  MPI_Win win;
  DNND2_CHECK_MPI(
      ::MPI_Win_create(base, count * disp_unit, disp_unit, info, comm, &win));
  MPI_Barrier(comm);

  DNND2_CHECK_MPI(::MPI_Win_lock_all(MPI_MODE_NOCHECK, win));

  return win;
}

void free_mpi_win(MPI_Win& win) {
  if (win == MPI_WIN_NULL) {
    return;
  }
  DNND2_CHECK_MPI(::MPI_Win_unlock_all(win));
  DNND2_CHECK_MPI(::MPI_Win_free(&win));
}

// void lock_win(const int lock_type, const int rank, MPI_Win& win) {
//   // Check win is not MPI_WIN_NULL
//   assert(win != MPI_WIN_NULL);
//   DNND2_CHECK_MPI(::MPI_Win_lock(lock_type, rank, 0, win));
// }

// void unlock_win(const int rank, MPI_Win& win) {
//   DNND2_CHECK_MPI(::MPI_Win_unlock(rank, win));
// }

// void one_sided_get(void* addr, const size_t count, MPI_Datatype data_type,
//                    const int target_rank, const size_t target_offset,
//                    MPI_Win& win) {
//   DNND2_CHECK_MPI(::MPI_Get(addr, count, data_type, target_rank,
//   target_offset,
//                             count, data_type, win));
// }

MPI_Request one_sided_rget(void* addr, const size_t count,
                           MPI_Datatype data_type, const int target_rank,
                           const size_t target_offset, MPI_Win& win) {
  MPI_Request request;
  DNND2_CHECK_MPI(::MPI_Rget(addr, count, data_type, target_rank, target_offset,
                             count, data_type, win, &request));
  return request;
}

void one_sided_wait(MPI_Request& request) {
  DNND2_CHECK_MPI(::MPI_Wait(&request, MPI_STATUS_IGNORE));
}

class rdm_comm {
 public:
  static constexpr size_t k_chunk_size = 1ULL << 26;

  rdm_comm(communicator& comm, const void* const base, const std::size_t size,
           MPI_Info info = MPI_INFO_NULL)
      : m_comm(comm) {
    priv_create(base, size, info);
  }

  void free() {
    for (auto& [rank, requests] : m_requests) {
      for (auto& request : requests) {
        if (request != MPI_REQUEST_NULL) {
          std::cerr << __FILE__ << ":" << __LINE__
                    << " Warning: pending request for rank " << rank
                    << " is not completed." << std::endl;
          one_sided_wait(request);
        }
      }
    }
    m_requests.clear();
    for (auto& win : m_sub_wins) {
      free_mpi_win(win);
    }
    m_sub_wins.clear();
  }

  // void fence() {
  //   for (auto& win : m_sub_wins) {
  //     win_fence(win);
  //   }
  // }

  // void lock(const int lock_type, const int rank) {
  //   for (auto& win : m_sub_wins) {
  //     lock_win(lock_type, rank, win);
  //   }
  // }

  // void unlock(const int rank) {
  //   for (auto& win : m_sub_wins) {
  //     unlock_win(rank, win);
  //   }
  // }

  void async_get(void* addr, const size_t size, const int target_rank) {
    size_t     offset            = 0;
    const auto num_chunks_to_get = (size + k_chunk_size - 1) / k_chunk_size;
    assert(num_chunks_to_get <= m_sub_wins.size());
    auto& requests = m_requests[target_rank];
    requests.assign(num_chunks_to_get, MPI_REQUEST_NULL);
    for (size_t i = 0; i < num_chunks_to_get; ++i) {
      const auto size_to_get = std::min(size - offset, k_chunk_size);
      requests[i]            = one_sided_rget(static_cast<char*>(addr) + offset,
                                              size_to_get, data_type::get<std::byte>(),
                                              target_rank, 0, m_sub_wins[i]);
      offset += size_to_get;
    }
  }

  void wait(const int target_rank) {
    auto it = m_requests.find(target_rank);
    if (it == m_requests.end()) {
      std::cerr << __FILE__ << ":" << __LINE__
                << " No pending request for rank " << target_rank << std::endl;
      return;
    }
    for (auto& request : it->second) {
      one_sided_wait(request);
    }
    m_requests.erase(it);
  }

 private:
  void priv_create(const void* const base, const std::size_t size,
                   MPI_Info info = MPI_INFO_NULL) {
    m_size          = size;
    auto num_chunks = (m_size + k_chunk_size - 1) / k_chunk_size;
    num_chunks      = m_comm.all_reduce_max(num_chunks);
    m_sub_wins.resize(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
      const auto offset = i * k_chunk_size;
      if (offset >= m_size) {
        // have to create a dummy window
        m_sub_wins[i] = create_mpi_win(nullptr, 0, 1, info, m_comm.comm());
      } else {
        const auto sub_count = std::min(m_size - offset, k_chunk_size);
        // DB
        // std::cerr << "Creating MPI Window: offset=" << offset
        //           << ", size=" << sub_count << std::endl;
        m_sub_wins[i] =
            create_mpi_win(static_cast<char*>(const_cast<void*>(base)) + offset,
                           sub_count, 1, info, m_comm.comm());
        // DB
        // std::cerr << "Created MPI Window: offset=" << offset
        //             << ", size=" << sub_count << std::endl;
      }
    }
  }

  communicator&                                     m_comm;
  size_t                                            m_size{0};
  std::vector<MPI_Win>                              m_sub_wins;
  std::unordered_map<int, std::vector<MPI_Request>> m_requests;
};

}  // namespace saltatlas::mpi