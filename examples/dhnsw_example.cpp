// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <ygm/comm.hpp>

#include <saltatlas/dhnsw/detail/hnswlib_space_wrapper.hpp>
#include <saltatlas/dhnsw/dhnsw.hpp>
#include <saltatlas/partitioner/voronoi_partitioner.hpp>

// User-defined distance function working on vectors of data
float my_l2_sqr(const std::vector<float> &x, const std::vector<float> &y) {
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

int main(int argc, char **argv) {
  // Initialize YGM
  ygm::comm world(&argc, &argv);
  int       mpi_rank = world.rank();
  int       mpi_size = world.size();

  // Example using my_l2_sqr distance function wrapped for use with hnswlib
  {
    if (mpi_rank == 0) {
      std::cout << "Distributed KNN example using user-defined distance "
                   "function with SpaceWrapper and vector data"
                << std::endl;
    }

    // Create wrapper around user-defined distance function. This allows
    // easy-to-write distance functions to be used with hnswlib
    auto my_l2_space = saltatlas::dhnsw_detail::SpaceWrapper(my_l2_sqr);

    using dist_t  = float;
    using index_t = std::size_t;
    using point_t = std::vector<float>;

    // Create partitioner
    saltatlas::voronoi_partitioner<dist_t, index_t, point_t> partitioner(
        world, my_l2_space);

    // Create indexing structure
    int              voronoi_rank = 2;
    int              num_seeds    = 2;
    saltatlas::dhnsw knn_index(voronoi_rank, num_seeds, &my_l2_space, &world,
                               partitioner);

    point_t              s0{-5.0, 0.0}, s1{5.0, 0.0};
    std::vector<point_t> seeds{s0, s1};
    partitioner.set_seeds(seeds);
    partitioner.fill_seed_hnsw();

    // Insert points from rank 0
    if (mpi_rank == 0) {
      // Define points to add to HNSW
      point_t p1{-4.0, 0.0}, p2{-5.0, 1.0}, p3{-6.0, 0.0}, p4{-5.0, -1.0};
      point_t p5{6.0, 0.0}, p6{5.0, 1.0}, p7{4.0, 0.0}, p8{5.0, -1.0};

      knn_index.queue_data_point_insertion(1, p1);
      knn_index.queue_data_point_insertion(2, p2);
      knn_index.queue_data_point_insertion(3, p3);
      knn_index.queue_data_point_insertion(4, p4);
      knn_index.queue_data_point_insertion(5, p5);
      knn_index.queue_data_point_insertion(6, p6);
      knn_index.queue_data_point_insertion(7, p7);
      knn_index.queue_data_point_insertion(8, p8);
    }

    // Set-up local HNSW's for data provided and insert local data into local
    // HNSW's
    knn_index.initialize_hnsw();

    // Lambda to execute upon completion of query
    auto report_lambda =
        [](const point_t                        &query_pt,
           const std::multimap<dist_t, index_t> &nearest_neighbors,
           auto                                  dhnsw) {
          std::cout << "Nearest neighbors for (" << query_pt[0] << ", "
                    << query_pt[1] << "): ";
          for (const auto &[dist, nearest_neighbor_index] : nearest_neighbors) {
            std::cout << nearest_neighbor_index << " ";
          }
          std::cout << std::endl;
        };

    // Default parameters to use with querying
    int k                   = 4;
    int query_hops          = 1;
    int query_voronoi_rank  = 2;
    int initial_num_queries = 1;

    // Points to query for
    point_t q1{-4.2, 1.1}, q2{5.4, 0.5}, q3{0.0, 0.0};

    // Spawn queries from different ranks
    if (mpi_rank == 0) {
      knn_index.query(q1, k, query_hops, initial_num_queries,
                      query_voronoi_rank, report_lambda);
      knn_index.query(q3, 2, query_hops, initial_num_queries,
                      query_voronoi_rank, report_lambda);
    } else if (mpi_rank == 1) {
      knn_index.query(q2, k, query_hops, initial_num_queries,
                      query_voronoi_rank, report_lambda);
    }

    knn_index.comm().barrier();
  }
  return 0;
}
