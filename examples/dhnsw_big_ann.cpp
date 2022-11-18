
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

#define DEFAULT_VORONOI_RANK 3
#define DEFAULT_NUM_HOPS 3
#define DEFAULT_NUM_INITIAL_QUERIES 1

using feature_t = uint16_t;
using point_t   = std::vector<feature_t>;
using dist_t    = float;
using index_t   = uint32_t;

dist_t l2_dist(const point_t& v1, const point_t& v2) {
  if (v1.size() != v2.size()) {
    std::cout << "Size mismatch: " << v1.size() << " != " << v2.size()
              << std::endl;
  }
  ASSERT_DEBUG(v1.size() == v2.size());

  dist_t d = 0;
  for (std::size_t i = 0; i < v1.size(); ++i) {
    d += (v2[i] - v1[i]) * (v2[i] - v1[i]);
  }

  return std::sqrt(d);
}

index_t calculate_line_offset(const std::vector<std::string>& filenames,
                              ygm::comm&                      world) {
  static index_t line_offset;
  static index_t line_count;
  line_offset = 0;
  line_count  = 0;

  ygm::io::line_parser linep(world, filenames);

  linep.for_all([](auto& line) { ++line_count; });

  world.barrier();

  struct recursive_offset_scan {
    void operator()(ygm::comm* c, index_t cumulative_size) {
      line_offset = cumulative_size;

      if (c->rank() < c->size() - 1) {
        c->async(c->rank() + 1, recursive_offset_scan(),
                 line_offset + line_count);
      }
    }
  };

  if (world.rank0() && world.size() > 1) {
    world.async(1, recursive_offset_scan(), line_count);
  }

  world.barrier();

  return line_offset;
}

ygm::container::pair_bag<index_t, point_t> read_points(
    const std::vector<std::string>& filenames, ygm::comm& world) {
  ygm::container::pair_bag<index_t, point_t> to_return(world);

  auto line_offset = calculate_line_offset(filenames, world);

  ygm::io::line_parser linep(world, filenames);

  index_t line_count{0};
  linep.for_all([&to_return, &line_offset, &line_count](auto& line) {
    point_t pt;

    std::stringstream ss(line);
    feature_t         feature;
    while (ss >> feature) {
      pt.push_back(feature);
    }

    to_return.async_insert(std::make_pair(line_offset + line_count, pt));
    ++line_count;
  });

  world.barrier();

  return to_return;
}

ygm::container::map<index_t, std::vector<index_t>> read_ground_truth(
    const std::vector<std::string>& filenames, ygm::comm& world, int k) {
  ygm::container::map<index_t, std::vector<index_t>> to_return(world);

  auto line_offset = calculate_line_offset(filenames, world);

  ygm::io::line_parser linep(world, filenames);

  index_t line_count{0};
  linep.for_all([&to_return, &line_offset, &line_count, k](auto& line) {
    std::vector<index_t> true_ngbrs;
    int                  ngbrs_found{0};

    std::stringstream ss(line);
    index_t           ngbr;
    while (ss >> ngbr && ngbrs_found < k) {
      true_ngbrs.push_back(ngbr);
      ++ngbrs_found;
    }

    to_return.async_insert(
        std::make_pair(line_offset + line_count, true_ngbrs));
    ++line_count;
  });

  world.barrier();

  return to_return;
}

void usage(ygm::comm& comm) {
  if (comm.rank0()) {
    std::cerr
        << "Usage: dhnsw_ascii_levenshtein -k <int> -s <int> [-v <int>] "
           "[-p <int>] -i <string>... -q <string>... -o <string>\n"
        << " -k <int>      - Number of nearest neighbors for querying\n"
        << " -s <int>   - Number of seeds (required)\n"
        << " -v <int>   - Voronoi rank (default is " << DEFAULT_VORONOI_RANK
        << ")\n"
        << " -p <int>   - Number of hops for querying (default is "
        << DEFAULT_NUM_HOPS << "\n"
        << " -n <int>		- Initial number of queries (default is "
        << DEFAULT_NUM_INITIAL_QUERIES << "\n"
        << " -i <string>... 	 - File(s) containing data to build index from "
           "(required)\n"
        << " -q <string>...   - File(s) containing data to query index with "
           "(required)\n"
        << " -g <string>		- File(s) containing ground truth for "
           "queries"
           "(required)\n"
        << " -h            - print help and exit\n\n";
  }
}

void parse_cmd_line(int argc, char** argv, ygm::comm& comm, int& voronoi_rank,
                    int& num_hops, int& num_seeds, int& k,
                    int&                      num_initial_queries,
                    std::vector<std::string>& index_filenames,
                    std::vector<std::string>& query_filenames,
                    std::vector<std::string>& ground_truth_filenames) {
  if (comm.rank0()) {
    std::cout << "CMD line:";
    for (int i = 0; i < argc; ++i) {
      std::cout << " " << argv[i];
    }
    std::cout << std::endl;
  }

  num_seeds                      = -1;
  k                              = -1;
  bool found_voronoi_rank        = false;
  bool found_num_hops            = false;
  bool found_initial_num_queries = false;

  int  c;
  bool inserting_index_filenames        = false;
  bool inserting_query_filenames        = false;
  bool inserting_ground_truth_filenames = false;
  bool prn_help                         = false;
  while (true) {
    while ((c = getopt(argc, argv, "+v:p:s:k:n:iqgh ")) != -1) {
      inserting_index_filenames        = false;
      inserting_query_filenames        = false;
      inserting_ground_truth_filenames = false;
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
        case 'q':
          inserting_query_filenames = true;
          break;
        case 'g':
          inserting_ground_truth_filenames = true;
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
    if (inserting_query_filenames) {
      query_filenames.push_back(argv[optind]);
    }
    if (inserting_ground_truth_filenames) {
      ground_truth_filenames.push_back(argv[optind]);
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

  // Detect misconfigured options
  if (index_filenames.size() < 1) {
    comm.cout0("Must specify file to build index from");
    prn_help = true;
  }
  if (query_filenames.size() < 1) {
    comm.cout0("Must specify file(s) to specifying queries");
    prn_help = true;
  }
  if (ground_truth_filenames.size() < 1) {
    comm.cout0("Must specify file(s) containing ground truth");
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
  std::vector<std::string> index_filenames;
  std::vector<std::string> query_filenames;
  std::vector<std::string> ground_truth_filenames;

  parse_cmd_line(argc, argv, world, voronoi_rank, num_hops, num_seeds, k,
                 num_initial_queries, index_filenames, query_filenames,
                 ground_truth_filenames);

  auto index_points = read_points(index_filenames, world);

  auto l2_space = saltatlas::dhnsw_detail::SpaceWrapper(l2_dist);

  saltatlas::metric_hyperplane_partitioner<dist_t, index_t, point_t>
      l2_partitioner(world, l2_space);
  // saltatlas::voronoi_partitioner<dist_t, index_t, point_t> fuzzy_partitioner(
  // world, fuzzy_leven_space);

  world.cout0("===Building nearest neighbor index===");

  saltatlas::dhnsw dist_index(voronoi_rank, num_seeds, &l2_space, &world,
                              l2_partitioner);

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

  auto query_points = read_points(query_filenames, world);
  auto ground_truth = read_ground_truth(ground_truth_filenames, world, k);
  auto num_queries  = query_points.size();

  world.cout0("===Beginning queries===");
  static std::vector<std::pair<index_t, std::vector<index_t>>> query_results;
  world.barrier();
  t.reset();

  auto store_results_lambda =
      [](const point_t&                        query_point,
         const std::multimap<dist_t, index_t>& nearest_neighbors,
         auto dist_knn_index, index_t query_ID) {
        std::pair<index_t, std::vector<index_t>> results;
        results.first = query_ID;

        for (const auto& dist_ID_pair : nearest_neighbors) {
          results.second.push_back(dist_ID_pair.second);
        }
      };

  query_points.for_all([&dist_index, k, num_hops, num_initial_queries,
                        voronoi_rank,
                        store_results_lambda](const auto& query_point) {
    dist_index.query(query_point.second, k, num_hops, num_initial_queries,
                     voronoi_rank, store_results_lambda, query_point.first);
  });

  world.barrier();

  auto elapsed = t.elapsed();
  world.cout0("Query time: ", elapsed);
  world.cout0("\t", num_queries / elapsed, " queries/sec");

  return 0;
}
