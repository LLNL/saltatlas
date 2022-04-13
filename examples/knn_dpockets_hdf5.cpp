// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <math.h>
#include <omp.h>
#include <saltatlas/saltatlas.hpp>
#include <saltatlas/utility.hpp>
#include <saltatlas_h5_io/h5_reader.hpp>
#include <saltatlas_h5_io/h5_writer.hpp>
#include <string>
#include <ygm/comm.hpp>
#include <ygm/utility.hpp>

// Doing this by hand because number of elements in vec may not be same as
// number of ranks. Not meant to be fast.
std::vector<size_t> postfix_sum(const std::vector<size_t> &in_vec,
                                ygm::comm                 &comm) {
  std::vector<size_t> to_return(in_vec.size());
  auto                to_return_ptr = comm.make_ygm_ptr(to_return);

  auto inserter_lambda = [](int i, size_t val, auto vec_ptr) {
    (*vec_ptr)[i + 1] = (*vec_ptr)[i] + val;
  };

  for (int i = 0; i < to_return.size() - 1; ++i) {
    if (i % comm.size() == comm.rank()) {
      for (int dest = 0; dest < comm.size(); ++dest) {
        comm.async(dest, inserter_lambda, i, in_vec[i], to_return_ptr);
      }
    }
    comm.barrier();
  }

  return to_return;
}

float my_l2_sqr(const std::vector<double> &x, const std::vector<double> &y) {
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

void read_data(
    const std::vector<std::string>                        &hdf_file_paths,
    saltatlas::dist_knn_index<float, std::vector<double>> &dist_index,
    const size_t                                           num_seeds) {
  std::vector<std::vector<double>> seed_features;
  seed_features.resize(num_seeds);

  // Make ygm pointer to seed_features vector
  auto seed_vector_ptr = dist_index.comm().make_ygm_ptr(seed_features);

  std::vector<size_t> my_file_sizes(hdf_file_paths.size());
  size_t              num_points{0};

  // Gather sizes of files each rank is responsible for
  for (int i = 0; i < hdf_file_paths.size(); ++i) {
    if (i % dist_index.comm().size() == dist_index.comm().rank()) {
      saltatlas::h5_io::h5_reader reader(hdf_file_paths[i]);

      if (!reader.is_open()) {
        std::cerr << "Failed to open" << std::endl;
        exit(EXIT_FAILURE);
      }

      std::vector<std::string> columns = {"col000", "col001", "col002",
                                          "col003", "col004", "col005",
                                          "col006", "col007"};
      const auto data = reader.read_columns_row_wise<double>(columns);

      my_file_sizes[i] = data.size();
      num_points += data.size();
    }
  }

  auto index_offsets = postfix_sum(my_file_sizes, dist_index.comm());
  num_points         = dist_index.comm().all_reduce_sum(num_points);
  index_offsets.push_back(num_points);

  // Determine seeds on rank 0 and distribute
  std::vector<size_t> seed_ids(num_seeds);
  if (dist_index.comm().rank0()) {
    std::cout << "Selecting seeds" << std::endl;
    saltatlas::utility::select_random_seed_ids(num_seeds, num_points, seed_ids);
    std::sort(seed_ids.begin(), seed_ids.end());
  }
  // TODO: Change to a ygm::bcast to avoid direct MPI call and use of
  // MPI_COMM_WORLD
  MPI_Bcast(seed_ids.data(), num_seeds, MPI_UNSIGNED_LONG_LONG, 0,
            MPI_COMM_WORLD);

  saltatlas::utility::fill_seed_vector_from_hdf5(seed_ids, hdf_file_paths,
                                                 index_offsets, seed_vector_ptr,
                                                 dist_index.comm());

  dist_index.comm().cout0("Creating HNSW from seeds");
  dist_index.set_seeds(seed_features);
  dist_index.fill_seed_hnsw();

  dist_index.comm().cout0("Distributing data across ranks");
  for (int i = 0; i < hdf_file_paths.size(); ++i) {
    if (24 * i % dist_index.comm().size() == dist_index.comm().rank()) {
      saltatlas::h5_io::h5_reader reader(hdf_file_paths[i]);

      if (!reader.is_open()) {
        std::cerr << "Failed to open" << std::endl;
        exit(EXIT_FAILURE);
      }

      std::vector<std::string> columns = {"col000", "col001", "col002",
                                          "col003", "col004", "col005",
                                          "col006", "col007"};
      const auto data = reader.read_columns_row_wise<double>(columns);
      for (int j = 0; j < data.size(); ++j) {
        auto data_id = index_offsets[i] + j;
        dist_index.queue_data_point_insertion(data_id, data[j]);
      }
    }
  }
  dist_index.comm().barrier();
}

void dpockets_query_trial(
    int voronoi_rank, int hops, std::string hdf_out_prefix,
    saltatlas::dist_knn_index<float, std::vector<double>> &dist_index) {
  if (dist_index.comm().rank() == 0) {
    std::cout << "Beginning query trial"
              << "\nVoronoi rank: " << voronoi_rank << "\nHops: " << hops
              << std::endl;
  }

  // Create HDF5 for storing results. Hard-coded to find 10 nearest neighbors
  // for each data point provided
  dist_index.comm().cout0("Opening output HDF5 files");
  std::string hdf_out_fname =
      hdf_out_prefix + std::to_string(dist_index.comm().rank());
  std::vector<std::string> hdf_column_names = {
      "index", "ngbr1", "ngbr2", "ngbr3", "ngbr4", "ngbr5",
      "ngbr6", "ngbr7", "ngbr8", "ngbr9", "ngbr10"};

  static saltatlas::h5_io::h5_writer writer(hdf_out_fname, hdf_column_names);

  if (!writer.is_open()) {
    std::cerr << "Failed to create HDF5 file" << std::endl;
  }

  // Static so lambda doesn't need to capture or make ygm_ptr objects to them
  static size_t correct_count;
  correct_count = 0;
  static size_t self_neighbors;
  self_neighbors = 0;
  static size_t total_neighbors;
  total_neighbors = 0;

  auto count_and_record_lambda =
      [](const std::vector<double>          &query_pt,
         const std::multimap<float, size_t> &nearest_neighbors,
         auto dist_knn_index, int index) {
        // Storing a row temporarily as a 1D matrix for use with HDF5 writer
        std::vector<std::vector<uint64_t>> row;
        row.resize(1);
        row[0].push_back(index);
        int correct_pocket = index / 10;
        for (const auto &[dist, nearest_neighbor_index] : nearest_neighbors) {
          row[0].push_back(nearest_neighbor_index);
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
        writer.write(row);
      };

  auto query_lambda = [&count_and_record_lambda, &dist_index, &hops,
                       &voronoi_rank](auto &index_pt) {
    dist_index.query(index_pt.second, 10, hops, 1, voronoi_rank,
                     count_and_record_lambda, index_pt.first);
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

    int num_seeds    = atoi(argv[1]);
    int voronoi_rank = atoi(argv[2]);
    int hops         = atoi(argv[3]);

    std::string hdf_out_prefix(argv[4]);

    std::vector<std::string> hdf_file_paths;
    for (int i = 5; i < argc; ++i) {
      hdf_file_paths.push_back((argv[i]));
    }

    world.cout0("Seeds: ", num_seeds, "\nVoronoi rank: ", voronoi_rank,
                "\nHops: ", hops, "\nOutput file prefix: ", hdf_out_prefix);

    auto my_l2_space = saltatlas::utility::SpaceWrapper(my_l2_sqr);
    saltatlas::dist_knn_index<float, std::vector<double>> dist_index(
        voronoi_rank, num_seeds, &my_l2_space, &world);

    world.cout0("Reading data files");
    step_timer.reset();
    read_data(hdf_file_paths, dist_index, num_seeds);
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

    step_timer.reset();
    dpockets_query_trial(voronoi_rank, hops, hdf_out_prefix, dist_index);
    world.barrier();
    world.cout0("Time: ", step_timer.elapsed(), "\n");
  }
}
