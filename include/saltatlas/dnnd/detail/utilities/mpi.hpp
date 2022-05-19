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

inline std::size_t distribute_tasks(const std::size_t num_local_tasks,
                                    const std::size_t batch_size,
                                    const int mpi_rank, const int mpi_size,
                                    const bool verbose) {
  // Gather the number of tasks to process to rank 0.
  std::size_t local_batch_size = 0;
  if (mpi_rank > 0) {
    SALTATLAS_DNND_CHECK_MPI(
        MPI_Send(&num_local_tasks, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD));

    MPI_Status status;
    SALTATLAS_DNND_CHECK_MPI(MPI_Recv(&local_batch_size, 1, MPI_UNSIGNED_LONG,
                                      0, 0, MPI_COMM_WORLD, &status));
  } else {
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
    std::size_t num_unassigned_tasks =
        (batch_size > 0) ? std::min(batch_size, num_global_tasks)
                         : num_global_tasks;

    std::vector<std::size_t> batch_size_table(mpi_size, 0);
    while (num_unassigned_tasks > 0) {
      const std::size_t max_num_tasks_per_rank =
          (num_unassigned_tasks + mpi_size - 1) / mpi_size;
      for (std::size_t r = 0; r < num_remaining_tasks_table.size(); ++r) {
        const auto n =
            std::min({max_num_tasks_per_rank, num_remaining_tasks_table[r],
                      num_unassigned_tasks});
        num_remaining_tasks_table[r] -= n;
        batch_size_table[r] += n;
        num_unassigned_tasks -= n;
      }
    }

    // Tell the computed numbers to the other ranks
    for (int r = 1; r < mpi_size; ++r) {
      SALTATLAS_DNND_CHECK_MPI(MPI_Send(
          &batch_size_table[r], 1, MPI_UNSIGNED_LONG, r, 0, MPI_COMM_WORLD));
    }
    local_batch_size = batch_size_table[0];

    if (verbose) {
      show_task_distribution(batch_size_table);
    }
  }

  return local_batch_size;
}

}  // namespace saltatlas::dndetail::mpi