#include <algorithm>
#include <random>
#include <vector>

#include <ygm/comm.hpp>

/**
 * @brief Finds iterators to elements smaller, larger, and equal to given
 *values
 *
 * @param vecs Vector of vectors of sorted values to compare against
 *  candidate_medians
 * @param candidate_medians Vector of candidate medians to compare values from
 *vecs against
 *
 * @return A vector of iterators to bounds of regions that are less than,
 *equal to, and greater than candidate medians
 **/
template <typename T>
std::vector<typename std::vector<T>::const_iterator> generate_separators(
    const std::vector<std::vector<T>> &vecs,
    const std::vector<T>              &candidate_medians,
    const std::vector<bool>           &median_found) {
  std::vector<typename std::vector<T>::const_iterator> to_return(2 *
                                                                 vecs.size());

  for (size_t i = 0; i < vecs.size(); ++i) {
    if (not median_found[i]) {
      const auto &cur_vec       = vecs[i];
      const auto &cur_candidate = candidate_medians[i];

      to_return[2 * i] =
          std::lower_bound(cur_vec.begin(), cur_vec.end(), cur_candidate);
      to_return[2 * i + 1] =
          std::upper_bound(cur_vec.begin(), cur_vec.end(), cur_candidate);
    }
  }

  return to_return;
}

/**
 * @brief Use sizes of splits to determine new candidate medians to test as
 *medians
 *
 * @param vecs Vector of vectors of sorted values to choose candidate medians
 *from
 * @param local_separators Iterators separating local values less than, equal
 *to, and greather than candidate medians
 *
 * @return A vector of new candidate medians
 **/
template <typename T>
void generate_candidate_medians(
    const std::vector<std::vector<T>> &vecs,
    const std::vector<typename std::vector<T>::const_iterator>
                                                         &local_separators,
    std::vector<typename std::vector<T>::const_iterator> &local_limits,
    std::vector<T> &current_medians, std::vector<bool> &median_found,
    ygm::comm &c) {
  // Calculate number of local points less than, equal to, and greater than
  // the current candidate medians
  std::vector<size_t> local_split_sizes(3 * vecs.size());
  for (int i = 0; i < vecs.size(); ++i) {
    if (not median_found[i]) {
      const auto &cur_vec = vecs[i];

      local_split_sizes[3 * i] =
          std::distance(cur_vec.begin(), local_separators[2 * i]);
      local_split_sizes[3 * i + 1] =
          std::distance(local_separators[2 * i], local_separators[2 * i + 1]);
      local_split_sizes[3 * i + 2] =
          std::distance(local_separators[2 * i + 1], cur_vec.end());
    }
  }

  // Get the global numbers of points less than, equal to, and greater than
  // the current candidate medians
  std::vector<size_t> global_split_sizes(local_split_sizes.size());
  MPI_Allreduce(local_split_sizes.data(), global_split_sizes.data(),
                local_split_sizes.size(), MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                c.get_mpi_comm());

  std::vector<size_t> local_num_candidates_vec(vecs.size());
  std::vector<size_t> global_num_candidates_vec(
      vecs.size() / c.size() + (c.rank() < vecs.size() % c.size()));
  for (int i = 0; i < vecs.size(); ++i) {
    if (not median_found[i]) {
      size_t total_vec_size = global_split_sizes[3 * i] +
                              global_split_sizes[3 * i + 1] +
                              global_split_sizes[3 * i + 2];
      if (global_split_sizes[3 * i] >
          total_vec_size / 2) {  // Median is smaller than current candidate
        local_limits[2 * i + 1] = local_separators[2 * i];
        current_medians[i]      = std::numeric_limits<T>::lowest();
      } else if (global_split_sizes[3 * i + 2] >
                 total_vec_size /
                     2) {  // Median is larger than current candidate
        local_limits[2 * i] = local_separators[2 * i + 1];
        current_medians[i]  = std::numeric_limits<T>::lowest();
      } else {  // Candidate is median
        median_found[i] = true;
      }
      local_num_candidates_vec[i] =
          std::distance(local_limits[2 * i], local_limits[2 * i + 1]);
    }
  }

  std::vector<size_t> local_candidate_prefix_sums(vecs.size());
  MPI_Exscan(local_num_candidates_vec.data(),
             local_candidate_prefix_sums.data(),
             local_num_candidates_vec.size(), MPI_UNSIGNED_LONG_LONG, MPI_SUM,
             c.get_mpi_comm());

  std::vector<size_t> new_candidate_indices(vecs.size(), 0);
  std::random_device  dev;
  std::mt19937_64     rng(dev());
  for (int i = 0; i < vecs.size(); ++i) {
    if (not median_found[i]) {
      MPI_Reduce(&local_num_candidates_vec[i],
                 &global_num_candidates_vec[i / vecs.size()], 1,
                 MPI_UNSIGNED_LONG_LONG, MPI_SUM, i % c.size(),
                 c.get_mpi_comm());
      if (i % c.size() == c.rank()) {
        std::uniform_int_distribution<size_t> dist(
            0, global_num_candidates_vec[i / vecs.size()] - 1);
        new_candidate_indices[i] = dist(rng);
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, new_candidate_indices.data(),
                new_candidate_indices.size(), MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                c.get_mpi_comm());

  for (int i = 0; i < vecs.size(); ++i) {
    if (not median_found[i]) {
      if (new_candidate_indices[i] >= local_candidate_prefix_sums[i] &&
          new_candidate_indices[i] <
              local_candidate_prefix_sums[i] + local_num_candidates_vec[i]) {
        size_t local_index =
            new_candidate_indices[i] - local_candidate_prefix_sums[i] +
            std::distance(vecs[i].begin(), local_limits[2 * i]);
        current_medians[i] = vecs[i][local_index];
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, current_medians.data(), current_medians.size(),
                ygm::detail::mpi_typeof(current_medians[0]), MPI_MAX,
                c.get_mpi_comm());
}

/**
 * @brief Compute a collection of medians. Each rank contains a vector of
 *vectors, the outer vectors all being the same length. The returned vector
 *contains medians for each of the inner vectors.
 *
 * @param vecs Vector of vectors of values to compute medians of. Data in
 *inner vectors is partitioned across ranks with no assumptions on the
 *distribution of values.
 *
 * @return A vector of median values
 **/
template <typename T>
std::vector<T> compute_medians(std::vector<std::vector<T>> &vecs,
                               ygm::comm                   &c) {
  // Sort vectors to begin. This makes the overall serial algorithm O(n
  // \log(n)) instead of O(n), but it makes things a bit easier
  for (size_t i = 0; i < vecs.size(); ++i) {
    std::sort(vecs[i].begin(), vecs[i].end());
  }

  std::vector<T> current_medians(vecs.size());
  std::vector<typename std::vector<T>::const_iterator> current_limits(
      2 * vecs.size());
  std::vector<bool> median_found(vecs.size(), false);
  int               num_medians_found{0};

  for (int i = 0; i < vecs.size(); ++i) {
    current_medians[i]        = std::numeric_limits<T>::lowest();
    current_limits[2 * i]     = vecs[i].begin();
    current_limits[2 * i + 1] = vecs[i].end();
  }

  int round = 0;
  while (num_medians_found < vecs.size() && round < 10) {
    auto local_separators =
        generate_separators(vecs, current_medians, median_found);

    generate_candidate_medians(vecs, local_separators, current_limits,
                               current_medians, median_found, c);

    num_medians_found = 0;
    for (int i = 0; i < vecs.size(); ++i) {
      num_medians_found += median_found[i];
    }

    ++round;
  }

  return current_medians;
}
