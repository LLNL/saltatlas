// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <math.h>
#include <unistd.h>
#include <string>

#include <ygm/comm.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/container/map.hpp>
#include <ygm/utility.hpp>

#include <saltatlas/dhnsw/detail/utility.hpp>
#include <saltatlas/dhnsw/dhnsw.hpp>
#include <saltatlas/partitioner/voronoi_partitioner.hpp>

template <typename FirstType, typename SecondType>
using pair_bag = ygm::container::bag<std::pair<FirstType, SecondType>>;

void usage(ygm::comm &comm) {
  if (comm.rank0()) {
    std::cerr
        << "Usage: -v <int> -h <int> -k <int> -i <string>\n"
        << " -v <int>   - Voronoi rank (required)\n"
        << " -s <int>   - Number of seeds (required)\n"
        << " -d <int>		- Number of dimensions in dataset (required)\n"
        << " -i <string> 	 - File containing data to build index from "
           "(required)\n"
        << " -h            - print help and exit\n\n";
  }
}

void parse_cmd_line(int argc, char **argv, ygm::comm &comm, int &voronoi_rank,
                    int &num_seeds, int &num_dimensions,
                    std::string &index_filename) {
  if (comm.rank0()) {
    std::cout << "CMD line:";
    for (int i = 0; i < argc; ++i) {
      std::cout << " " << argv[i];
    }
    std::cout << std::endl;
  }

  bool found_index_filename{false};

  voronoi_rank   = -1;
  num_seeds      = -1;
  num_dimensions = -1;
  index_filename = "";

  int  c;
  bool prn_help = false;
  while ((c = getopt(argc, argv, "v:s:d:i:h ")) != -1) {
    switch (c) {
      case 'h':
        prn_help = true;
        break;
      case 'v':
        voronoi_rank = atoi(optarg);
        break;
      case 's':
        num_seeds = atoi(optarg);
        break;
      case 'd':
        num_dimensions = atoi(optarg);
        break;
      case 'i':
        found_index_filename = true;
        index_filename       = optarg;
        break;
      default:
        std::cerr << "Unrecognized option: " << c << ", ignore." << std::endl;
        prn_help = true;
        break;
    }
  }

  // Detect misconfigured options
  if (!found_index_filename) {
    comm.cout0("Must specify file to build index from");
    prn_help = true;
  }
  if (voronoi_rank < 0) {
    comm.cout0(
        "Must specify positive Voronoi rank to build distributed index with");
    prn_help = true;
  }
  if (num_seeds < 1) {
    comm.cout0(
        "Must specify a positive number of seeds to use for building "
        "distributed index");
    prn_help = true;
  }
  if (num_dimensions < 0) {
    comm.cout0("Must specify a positive number of dimensions in dataset");
    prn_help = true;
  }

  if (prn_help) {
    usage(comm);
    exit(-1);
  }
}

float my_cos_sim_squared(const std::vector<float> &x,
                         const std::vector<float> &y) {
  if (x.size() != y.size()) {
    std::cerr << "Size mismatch for l2 distance" << std::endl;
    exit;
  }

  float x_magnitude{0.0};
  float y_magnitude{0.0};
  float dot_product{0.0};

  for (size_t i = 0; i < x.size(); ++i) {
    x_magnitude += x[i] * x[i];
    y_magnitude += y[i] * y[i];
    dot_product += x[i] * y[i];
  }

  return dot_product * dot_product / (x_magnitude * y_magnitude);
}

template <typename IndexType, typename Point>
pair_bag<IndexType, Point> read_data(
    ygm::container::bag<std::string> &bag_filenames, int num_dimensions) {
  pair_bag<IndexType, Point> to_return(bag_filenames.comm());

  auto read_file_lambda = [&to_return, &num_dimensions](const auto &fname) {
    std::ifstream ifs(fname.c_str());
    std::string   line;
    while (std::getline(ifs, line)) {
      std::stringstream  ssline(line);
      size_t             point_index;
      float              val;
      std::string        binary_name;
      std::vector<float> point;
      // ssline >> point_index;
      // ssline >> binary_name;

      std::string point_index_str;
      std::getline(ssline, point_index_str, '\t');
      std::getline(ssline, binary_name, '\t');

      point_index = std::stoll(point_index_str);

      std::string val_str;
      while (std::getline(ssline, val_str, '\t') &&
             (point.size() < num_dimensions)) {
        try {
          point.push_back(stof(val_str));
        } catch (...) {
          std::cout << "Error processing value: " << val_str << " in line "
                    << line << std::endl;
        }
      }
      to_return.async_insert(std::make_pair(point_index, point));
    }
  };

  bag_filenames.for_all(read_file_lambda);

  bag_filenames.comm().barrier();

  return to_return;
}

template <typename DistType, typename IndexType, typename Point,
          template <typename, typename, typename> class Partitioner>
void build_index(
    pair_bag<IndexType, Point>                                &bag_data,
    saltatlas::dhnsw<DistType, IndexType, Point, Partitioner> &dist_index,
    const size_t num_seeds, const int num_dimensions) {
  if (dist_index.comm().rank0()) {
    std::cout << "\n****Building distributed index****"
              << "\nNumber of Voronoi cells: " << num_seeds << std::endl;
  }

  ygm::timer step_timer{};

  dist_index.comm().cout0("Distributing data across ranks");

  auto add_point_lambda = [&dist_index, &bag_data](const auto &index,
                                                   const auto &point) {
    dist_index.queue_data_point_insertion(index, point);
  };

  dist_index.comm().barrier();
  step_timer.reset();
  bag_data.for_all(add_point_lambda);

  dist_index.comm().barrier();
  dist_index.comm().cout0("Data distribution time: ", step_timer.elapsed());
  step_timer.reset();

  dist_index.comm().cout0("Initializing per-cell HNSW structures");
  dist_index.initialize_hnsw();
  // dist_index.print_insertion_queue_sizes();
  dist_index.comm().barrier();
  dist_index.comm().cout0("HNSW initialization time: ", step_timer.elapsed());

  size_t global_hnsw_size = dist_index.global_size();
  dist_index.comm().cout0("Global HNSW size: ", global_hnsw_size);

  dist_index.comm().cout0("Finished creating index structure");
}

void fill_filenames_bag(ygm::container::bag<std::string> &bag,
                        const std::string                &filename) {
  // Test file to see if it contains filenames
  if (bag.comm().rank0()) {
    std::ifstream ifs(filename.c_str());
    std::string   line;

    while (std::getline(ifs, line)) {
      std::ifstream csv_ifs(line);

      if (csv_ifs.is_open()) {
        // bag.comm().cout() << "Input file found: " << line << std::endl;
        bag.async_insert(line);
      } else {
        bag.comm().cout() << "File: " << line
                          << " is not readable. Assuming input is CSV."
                          << std::endl;
        bag.async_insert(filename);
        break;
      }
    }
  }

  bag.comm().barrier();

  return;
}

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);
  {
    ygm::timer step_timer{};

    int mpi_rank = world.rank();
    int mpi_size = world.size();

    int voronoi_rank;
    int hops;
    int num_seeds;
    int k;
    int num_dimensions;

    std::string index_filename;

    parse_cmd_line(argc, argv, world, voronoi_rank, num_seeds, num_dimensions,
                   index_filename);

    ygm::container::bag<std::string> bag_index_filenames(world);

    fill_filenames_bag(bag_index_filenames, index_filename);

    using dist_t  = float;
    using index_t = std::size_t;
    using point_t = std::vector<float>;

    auto bag_data =
        read_data<index_t, point_t>(bag_index_filenames, num_dimensions);

    uint64_t num_points = bag_data.size();

    auto my_space = saltatlas::dhnsw_detail::SpaceWrapper(my_cos_sim_squared);

    saltatlas::voronoi_partitioner<dist_t, index_t, point_t> partitioner(
        world, my_space);

    // Build index
    saltatlas::dhnsw dist_index(voronoi_rank, num_seeds, &my_space, &world,
                                partitioner);

    dist_index.partition_data(bag_data, num_seeds);

    step_timer.reset();
    build_index(bag_data, dist_index, num_seeds, num_dimensions);
    dist_index.comm().barrier();
    world.cout0("Total index construction time: ", step_timer.elapsed());
  }
}
