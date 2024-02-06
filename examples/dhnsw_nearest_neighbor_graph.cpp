
#include <unistd.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <saltatlas/container/pair_bag.hpp>
#include <saltatlas/dhnsw/detail/utility.hpp>
#include <saltatlas/dhnsw/dhnsw.hpp>
#include <saltatlas/partitioner/metric_hyperplane_partitioner.hpp>
#include <saltatlas/partitioner/voronoi_partitioner.hpp>

#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/container/set.hpp>
#include <ygm/io/line_parser.hpp>
#include <ygm/utility.hpp>

#include <boost/json/src.hpp>

#define DEFAULT_VORONOI_RANK 3
#define DEFAULT_NUM_HOPS 3
#define DEFAULT_NUM_INITIAL_QUERIES 2
#define DEFAULT_HNSW_M 16
#define DEFAULT_HNSW_EF_CONSTRUCTION 100

using feature_t = uint16_t;
using point_t   = std::vector<feature_t>;
using dist_t    = float;
using index_t   = uint32_t;

dist_t l2_sqr(const point_t& v1, const point_t& v2) {
  if (v1.size() != v2.size()) {
    std::cout << "Size mismatch: " << v1.size() << " != " << v2.size()
              << std::endl;
  }
  ASSERT_DEBUG(v1.size() == v2.size());

  dist_t d = 0;
  for (std::size_t i = 0; i < v1.size(); ++i) {
    d += (v2[i] - v1[i]) * (v2[i] - v1[i]);
  }

  return d;
}

ygm::container::pair_bag<index_t, point_t> read_points(
    const std::vector<std::string>& filenames, ygm::comm& world) {
  ygm::container::pair_bag<index_t, point_t> to_return(world);

  ygm::io::line_parser linep(world, filenames);

  linep.for_all([&to_return](auto& line) {
    point_t pt;

    std::stringstream ss(line);

    index_t id;
    ss >> id;

    feature_t feature;
    while (ss >> feature) {
      pt.push_back(feature);
    }

    to_return.async_insert(std::make_pair(id, pt));
  });

  world.barrier();

  return to_return;
}

void usage(ygm::comm& comm) {
  if (comm.rank0()) {
    std::cerr
        << "Usage: dhnsw_nearest_neighbor_graph -k <int> -s <int> [-v <int>] "
           "[-p <int>] -i <string>... -o <string>\n"
        << " -k <int>      			- Number of nearest neighbors "
           "for querying\n"
        << " -s <int>   				- Number of seeds "
           "(required)\n"
        << " -v <int>   				- Voronoi rank "
           "(default is "
        << DEFAULT_VORONOI_RANK << ")\n"
        << " -p <int>   				- Number of hops for "
           "querying (default is "
        << DEFAULT_NUM_HOPS << "\n"
        << " -n <int>						- Initial "
           "number of queries (default is "
        << DEFAULT_NUM_INITIAL_QUERIES << "\n"
        << " -i <string>...			- File(s) containing data to "
           "build index from "
           "(required)\n"
        << " -o <string>				- Output directory for "
           "nearest neighbor graph (required)\n"
        << " -m <int>						- Internal "
           "HNSWlib m parameter (default is "
        << DEFAULT_HNSW_M << "\n"
        << " -e <int>						- Internal "
           "HNSWlib ef_construction parameter "
           "(default is "
        << DEFAULT_HNSW_EF_CONSTRUCTION << "\n"
        << " -h            			- print help and exit\n\n";
  }
}

void parse_cmd_line(int argc, char** argv, ygm::comm& comm, int& voronoi_rank,
                    int& num_hops, int& num_seeds, int& k,
                    int& num_initial_queries, int& hnsw_m,
                    int&                      hnsw_ef_construction,
                    std::vector<std::string>& index_filenames,
                    std::string&              output_directory) {
  if (comm.rank0()) {
    std::cout << "CMD line:";
    for (int i = 0; i < argc; ++i) {
      std::cout << " " << argv[i];
    }
    std::cout << std::endl;
  }

  num_seeds                       = -1;
  bool found_voronoi_rank         = false;
  bool found_num_hops             = false;
  bool found_initial_num_queries  = false;
  bool found_hnsw_m               = false;
  bool found_hnsw_ef_construction = false;

  int  c;
  bool inserting_index_filenames = false;
  bool prn_help                  = false;
  while (true) {
    while ((c = getopt(argc, argv, "+v:p:s:k:n:io:m:e:h ")) != -1) {
      inserting_index_filenames = false;
      switch (c) {
        case 'h':
          prn_help = true;
          break;
        case 'v':
          voronoi_rank       = atoi(optarg);
          found_voronoi_rank = true;
          break;
        case 's':
          num_seeds = atoi(optarg);
          break;
        case 'p':
          num_hops       = atoi(optarg);
          found_num_hops = true;
          break;
        case 'k':
          k = atoi(optarg);
          break;
        case 'n':
          num_initial_queries       = atoi(optarg);
          found_initial_num_queries = true;
          break;
        case 'i':
          inserting_index_filenames = true;
          break;
        case 'o':
          output_directory = optarg;
          break;
        case 'm':
          hnsw_m       = atoi(optarg);
          found_hnsw_m = true;
          break;
        case 'e':
          hnsw_ef_construction       = atoi(optarg);
          found_hnsw_ef_construction = true;
          break;
        default:
          std::cerr << "Unrecognized option: " << c << ", ignore." << std::endl;
          prn_help = true;
          break;
      }
    }
    if (optind >= argc) break;

    if (inserting_index_filenames) {
      index_filenames.push_back(argv[optind]);
    }

    ++optind;
  }

  if (!found_voronoi_rank) {
    comm.cout0("Using default voronoi rank: ", DEFAULT_VORONOI_RANK);
    voronoi_rank = DEFAULT_VORONOI_RANK;
  }
  if (!found_num_hops) {
    comm.cout0("Using default number of hops: ", DEFAULT_NUM_HOPS);
    num_hops = DEFAULT_NUM_HOPS;
  }
  if (!found_initial_num_queries) {
    comm.cout0("Using default number of initial queries: ",
               DEFAULT_NUM_INITIAL_QUERIES);
    num_initial_queries = DEFAULT_NUM_INITIAL_QUERIES;
  }
  if (!found_hnsw_m) {
    comm.cout0("Using default hnsw_m: ", DEFAULT_HNSW_M);
  }
  if (!found_hnsw_ef_construction) {
    comm.cout0("Using default hnsw_ef_construction: ",
               DEFAULT_HNSW_EF_CONSTRUCTION);
  }

  // Detect misconfigured options
  if (index_filenames.size() < 1) {
    comm.cout0("Must specify file to build index from");
    prn_help = true;
  }
  if (output_directory.size() < 1) {
    comm.cout0("Must specify output directory");
    prn_help = true;
  }
  if (num_seeds < 1) {
    comm.cout0(
        "Must specify a positive number of seeds to use for building "
        "distributed index");
    prn_help = true;
  }
  if (k < 1) {
    comm.cout0(
        "Must specify positive number of nearest neighbors to query index "
        "for");
    prn_help = true;
  }

  if (prn_help) {
    usage(comm);
    exit(-1);
  }
}

int main(int argc, char** argv) {
  ygm::comm world(&argc, &argv);

  int                      voronoi_rank;
  int                      num_hops;
  int                      num_seeds;
  int                      k;
  int                      num_initial_queries;
  int                      hnsw_m;
  int                      hnsw_ef_construction;
  std::vector<std::string> index_filenames;
  std::string              output_directory;

  parse_cmd_line(argc, argv, world, voronoi_rank, num_hops, num_seeds, k,
                 num_initial_queries, hnsw_m, hnsw_ef_construction,
                 index_filenames, output_directory);

  auto index_points = read_points(index_filenames, world);

  auto l2_space = saltatlas::dhnsw_detail::SpaceWrapper(l2_sqr);

  saltatlas::metric_hyperplane_partitioner<dist_t, index_t, point_t>
      l2_partitioner(world, l2_space);
  // saltatlas::voronoi_partitioner<dist_t, index_t, point_t> fuzzy_partitioner(
  // world, fuzzy_leven_space);

  world.cout0("===Building nearest neighbor index===");

  saltatlas::dhnsw dist_index(voronoi_rank, num_seeds, &l2_space, &world,
                              l2_partitioner);

  saltatlas::dhnsw_detail::hnsw_params_t hnsw_params;
  hnsw_params.M               = hnsw_m;
  hnsw_params.ef_construction = hnsw_ef_construction;
  dist_index.set_hnsw_params(hnsw_params);

  world.barrier();
  ygm::timer t{};

  dist_index.partition_data(index_points, num_seeds);

  world.barrier();

  world.cout0("Distributing data to local HNSWs");
  index_points.for_all([&dist_index, &world](const auto& ID_point) {
    dist_index.queue_data_point_insertion(ID_point.first, ID_point.second);
  });

  world.barrier();

  world.cout0("Initializing local HNSWs");
  dist_index.initialize_hnsw();

  world.barrier();
  world.cout0("Finished nearest neighbor index construction");
  world.cout0("Total build time: ", t.elapsed());
  world.cout0("Global HNSW size: ", dist_index.global_size());

  static std::ofstream rank_fstream(output_directory +
                                    "/nearest_neighbor_graph" +
                                    std::to_string(world.rank()));

  auto store_results_lambda =
      [](const point_t&                        query_point,
         const std::multimap<dist_t, index_t>& nearest_neighbors,
         auto dist_knn_index, index_t query_ID) {
        boost::json::object obj;
        obj.emplace("ID", query_ID);

        boost::json::array features;
        for (const auto& feature : query_point) {
          features.emplace_back(feature);
        }
        obj.emplace("Features", features);

        boost::json::array neighbors;
        for (const auto& dist_ngbr : nearest_neighbors) {
          neighbors.emplace_back(dist_ngbr.second);
        }
        obj.emplace("Neighbors", neighbors);

        rank_fstream << obj << "\n";
      };

  world.cout0("Querying for nearest neighbor graph");

  dist_index.for_all_data([&dist_index, k, num_hops, num_initial_queries,
                           voronoi_rank,
                           store_results_lambda](const auto& query_point) {
    dist_index.query(query_point.second, k, num_hops, num_initial_queries,
                     voronoi_rank, store_results_lambda, query_point.first);
  });

  world.barrier();

  auto elapsed = t.elapsed();

  return 0;
}
