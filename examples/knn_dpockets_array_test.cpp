// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <math.h>
#include <omp.h>
#include <saltatlas/saltatlas.hpp>
#include <string>
#include <ygm/comm.hpp>
#include <ygm/utility.hpp>

float my_l2_sqr(const std::array<float, 8> &x, const std::array<float, 8> &y) {
  if (x.size() != y.size()) {
    std::cerr << "Size mismatch for l2 distance" << std::endl;
    exit;
  }

  float dist_sqr{0.0};

  for (size_t i = 0; i < x.size(); ++i) {
    dist_sqr += (x[i] - y[i]) * (x[i] - y[i]);
  }

  return dist_sqr;
}

size_t count_points(const std::vector<std::string> &fnames, ygm::comm &comm) {
  size_t local_num_lines{0};

  int mpi_rank = comm.rank();
  int mpi_size = comm.size();

  for (int i = 0; i < fnames.size(); ++i) {
    if (i % mpi_size == mpi_rank) {
      std::ifstream ifs(fnames[i].c_str());
      ifs.imbue(std::locale::classic());
      std::string line;

      while (std::getline(ifs, line)) {
        ++local_num_lines;
      }
    }
  }

  return comm.all_reduce_sum(local_num_lines);
}

void select_seed_ids(const int num_seeds, const size_t num_points,
                     std::vector<size_t> &seed_ids) {
  std::mt19937                          rng(123);
  std::uniform_int_distribution<size_t> uni(0, num_points - 1);

  seed_ids.clear();
  seed_ids.reserve(num_seeds);

  for (int i = 0; i < num_seeds; ++i) {
    seed_ids.push_back(uni(rng));
  }

  std::sort(seed_ids.begin(), seed_ids.end());
  seed_ids.erase(std::unique(seed_ids.begin(), seed_ids.end()), seed_ids.end());

  while (seed_ids.size() < num_seeds) {
    // std::cout << "Replacing duplicate seed_ids" << std::endl;
    int s = seed_ids.size();
    for (int i = 0; i < num_seeds - s; ++i) {
      seed_ids.push_back(uni(rng));
    }
    std::sort(seed_ids.begin(), seed_ids.end());
    seed_ids.erase(std::unique(seed_ids.begin(), seed_ids.end()),
                   seed_ids.end());
  }

  assert(seed_ids.size() == num_seeds);

  return;
}

template <typename FEATURES>
void fill_seed_vector(const std::vector<size_t>      &seed_ids,
                      const std::vector<std::string> &fnames,
                      std::vector<FEATURES> &seed_features, ygm::comm &comm) {
  seed_features.resize(seed_ids.size());

  int mpi_size = comm.size();
  int mpi_rank = comm.rank();

  int curr_seed_ptr{0};

  const int coordinator_rank{0};
  auto      store_seed_features_lambda =
      [](auto comm, size_t seed_index, std::array<float, 8> vals,
         auto seeds_vector_ptr) { (*seeds_vector_ptr)[seed_index] = vals; };

  // Make ygm pointer to seed_features vector
  auto seeds_vector_ptr = comm.make_ygm_ptr(seed_features);

  for (size_t i = 0; i < fnames.size(); ++i) {
    if (i % mpi_size == mpi_rank) {
      std::ifstream ifs(fnames[i].c_str());
      ifs.imbue(std::locale::classic());
      std::string line;

      // Hard-coded to read dpockets data
      while (std::getline(ifs, line) && curr_seed_ptr < seed_ids.size()) {
        std::stringstream ssline(line);
        size_t            index;
        float             val;

        ssline >> index;
        while (index > seed_ids[curr_seed_ptr] &&
               curr_seed_ptr < seed_ids.size()) {
          ++curr_seed_ptr;
        }
        if (index == seed_ids[curr_seed_ptr]) {
          float                val;
          std::array<float, 8> vals;
          // Hard-code 8 features
          for (size_t i = 0; i < 8; ++i) {
            ssline >> val;
            vals[i] = val;
          }
          // Send all seeds to rank 0 initially
          comm.async(coordinator_rank, store_seed_features_lambda,
                     curr_seed_ptr, vals, seeds_vector_ptr);
          curr_seed_ptr++;
        }
      }
    }
  }

  comm.barrier();

  // Broadcast all seeds from rank 0 (in the worst way possible...)
  // This is made more difficult by storing seeds as a vector of arrays
  if (mpi_rank == coordinator_rank) {
    for (size_t i = 0; i < seed_features.size(); ++i) {
      for (int dest = 0; dest < mpi_size; ++dest) {
        if (dest == coordinator_rank) continue;
        comm.async(dest, store_seed_features_lambda, i, seed_features[i],
                   seeds_vector_ptr);
      }
    }
  }

  comm.barrier();

  return;
}

void read_data(
    const std::vector<std::string>                         &fnames,
    saltatlas::dist_knn_index<float, std::array<float, 8>> &dist_index,
    const size_t                                            num_seeds) {
  int mpi_rank = dist_index.comm().rank();
  int mpi_size = dist_index.comm().size();

  std::vector<std::pair<size_t, std::array<float, 8>>> data_vec;

  size_t num_points = count_points(fnames, dist_index.comm());

  // Create seed IDs
  std::vector<size_t> seed_ids;
  if (mpi_rank == 0) {
    select_seed_ids(num_seeds, num_points, seed_ids);
  } else {
    seed_ids.resize(num_seeds);
  }
  // TODO: Fix this... Can't access comm like old YGM, so hard-coded
  // MPI_COMM_WORLD
  MPI_Bcast(seed_ids.data(), num_seeds * sizeof(decltype(seed_ids)::value_type),
            MPI_BYTE, 0, MPI_COMM_WORLD);

  // Read seed features
  std::vector<std::array<float, 8>> seed_features;
  fill_seed_vector(seed_ids, fnames, seed_features, dist_index.comm());

  dist_index.set_seeds(seed_features);
  dist_index.fill_seed_hnsw();

  // Read all sample features
  // (Using Geoff's example, hardcoded to 8 features)
  for (size_t i = 0; i < fnames.size(); ++i) {
    if (i % mpi_size == mpi_rank) {
      std::ifstream ifs(fnames[i].c_str());
      std::string   line;
      while (std::getline(ifs, line)) {
        std::stringstream    ssline(line);
        size_t               index;
        float                val;
        std::array<float, 8> values;
        ssline >> index;
        for (size_t i = 0; i < 8; ++i) {
          ssline >> val;
          values[i] = val;
        }
        data_vec.push_back(std::make_pair(index, std::move(values)));
      }
    }
  }

  for (auto &data_point : data_vec) {
    dist_index.queue_data_point_insertion(data_point.first, data_point.second);
  }
  dist_index.comm().barrier();
}

std::vector<int> make_voronoi_ranks_vec(int min_voronoi_rank,
                                        int max_voronoi_rank) {
  std::vector<int> to_return;

  for (int i = min_voronoi_rank; i <= max_voronoi_rank; ++i) {
    to_return.push_back(i);
  }

  return std::move(to_return);
}

std::vector<int> make_hops_vec(int min_hops, int max_hops) {
  std::vector<int> to_return;

  if (min_hops == 0) {
    to_return.push_back(0);
    min_hops = 1;
  }

  for (int i = min_hops; i <= max_hops; i *= 2) {
    to_return.push_back(i);
  }

  return std::move(to_return);
}

void dpockets_query_trial(
    int voronoi_rank, int hops,
    saltatlas::dist_knn_index<float, std::array<float, 8>> &dist_index) {
  dist_index.comm().cout0("Beginning query trial",
                          "\nVoronoi rank: ", voronoi_rank, "\nHops: ", hops);

  // Static so lambda doesn't need to capture or make ygm_ptr objects to them
  static size_t correct_count;
  correct_count = 0;
  static size_t self_neighbors;
  self_neighbors = 0;
  static size_t total_neighbors;
  total_neighbors = 0;

  auto count_lambda = [](const std::array<float, 8>         &query_pt,
                         const std::multimap<float, size_t> &nearest_neighbors,
                         auto dist_knn_index, int index) {
    int correct_pocket = index / 10;
    for (const auto &[dist, nearest_neighbor_index] : nearest_neighbors) {
      if (nearest_neighbor_index / 10 == correct_pocket &&
          nearest_neighbor_index != index) {
        ++correct_count;
      } else {
        if (nearest_neighbor_index == index) {
          ++self_neighbors;
        }
      }
      ++total_neighbors;
    }
  };

  auto query_lambda = [&count_lambda, &dist_index, &hops,
                       &voronoi_rank](auto &index_pt) {
    dist_index.query(index_pt.second, 10, hops, 1, voronoi_rank, count_lambda,
                     index_pt.first);
  };

  dist_index.for_all_data(query_lambda);

  dist_index.comm().barrier();

  size_t global_correct = dist_index.comm().all_reduce_sum(correct_count);
  size_t global_self_neighbors =
      dist_index.comm().all_reduce_sum(self_neighbors);
  size_t global_neighbors = dist_index.comm().all_reduce_sum(total_neighbors);
  dist_index.comm().cout0("Self neighbors: ", global_self_neighbors,
                          "\nTotal neighbors: ", global_neighbors,
                          "\nCorrect neighbors: ", global_correct);

  return;
}

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);
  {
    ygm::timer step_timer{};

    int mpi_rank = world.rank();
    int mpi_size = world.size();

    int num_seeds        = atoi(argv[1]);
    int min_voronoi_rank = atoi(argv[2]);
    int max_voronoi_rank = atoi(argv[3]);
    int min_hops         = atoi(argv[4]);
    int max_hops         = atoi(argv[5]);

    std::vector<std::string> input_fnames;
    for (size_t i = 5; i < argc; ++i) {
      input_fnames.push_back(std::string(argv[i]));
    }

    world.cout0("Seeds: ", num_seeds,
                "\nMinimum Voronoi rank: ", min_voronoi_rank,
                "\nMaximum Voronoi rank: ", max_voronoi_rank,
                "\nMinimum Hops: ", min_hops, "\nMaximum Hops: ", max_hops);

    std::vector<int> voronoi_ranks_vec =
        make_voronoi_ranks_vec(min_voronoi_rank, max_voronoi_rank);
    std::vector<int> hops_vec = make_hops_vec(min_hops, max_hops);

    auto             my_l2_space = saltatlas::utility::SpaceWrapper(my_l2_sqr);
    hnswlib::L2Space their_l2_space(8);
    saltatlas::dist_knn_index<float, std::array<float, 8>> dist_index(
        max_voronoi_rank, num_seeds, &my_l2_space, &world);

    world.cout0("Reading data files");
    step_timer.reset();
    read_data(input_fnames, dist_index, num_seeds);
    dist_index.comm().barrier();
    world.cout0("Read data time: ", step_timer.elapsed());

    world.cout0("Initializing per-cell HNSW structures");
    step_timer.reset();
    dist_index.initialize_hnsw();
    dist_index.comm().barrier();
    world.cout0("HNSW initialization time: ", step_timer.elapsed());

    size_t global_hnsw_size = dist_index.global_size();
    world.cout0("Global HNSW size: ", global_hnsw_size);

    world.cout0("Finished creating indexing structure",
                "\nBeginning query trials\n");

    for (int rank : voronoi_ranks_vec) {
      for (int hops : hops_vec) {
        step_timer.reset();
        dpockets_query_trial(rank, hops, dist_index);
        world.barrier();
        world.cout0("Time: ", step_timer.elapsed(), "\n");
      }
    }
  }
}
