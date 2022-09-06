// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include <mpi.h>

#define SALTATLAS_DNND_CHECK_MPI(ret)                                         \
  do {                                                                        \
    if (ret != MPI_SUCCESS) {                                                 \
      std::cerr << __FILE__ << ":" << __LINE__ << " MPI error." << std::endl; \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                \
    }                                                                         \
  } while (0)

namespace saltatlas::dndetail::mpi {

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
/// \param batch_size Global batch size. Up to this number of tasks are assigned
/// over all ranks. If 0 is specified, all tasks are assigned.
/// \param mpi_rank My MPI rank.
/// \param mpi_size MPI size.
/// \param verbose Verbose mode.
/// \return #of tasks assigned to myself.
inline std::size_t assign_tasks(const std::size_t num_local_tasks,
                                const std::size_t batch_size,
                                const int mpi_rank, const int mpi_size,
                                const bool verbose) {
  if (batch_size == 0) {
    return num_local_tasks;
  }

  // Gather the number of tasks to process to rank 0.
  std::size_t local_num_assigned_tasks = 0;
  if (mpi_rank > 0) {
    SALTATLAS_DNND_CHECK_MPI(
        MPI_Send(&num_local_tasks, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD));

    MPI_Status status;
    SALTATLAS_DNND_CHECK_MPI(MPI_Recv(&local_num_assigned_tasks, 1,
                                      MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD,
                                      &status));
  } else {
    // Gather the number of tasks each MPI has
    std::vector<std::size_t> num_remaining_tasks_table(mpi_size, 0);
    num_remaining_tasks_table[0] = num_local_tasks;
    for (int r = 1; r < mpi_size; ++r) {
      MPI_Status status;
      SALTATLAS_DNND_CHECK_MPI(MPI_Recv(&num_remaining_tasks_table[r], 1,
                                        MPI_UNSIGNED_LONG, r, 0, MPI_COMM_WORLD,
                                        &status));
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
          (num_global_unassigned_tasks < mpi_size)
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
      SALTATLAS_DNND_CHECK_MPI(MPI_Send(&task_assignment_table[r], 1,
                                        MPI_UNSIGNED_LONG, r, 0,
                                        MPI_COMM_WORLD));
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

  return local_num_assigned_tasks;
}

}  // namespace saltatlas::dndetail::mpi