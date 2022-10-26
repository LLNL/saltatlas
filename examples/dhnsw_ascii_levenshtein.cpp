
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

#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/container/set.hpp>
#include <ygm/io/line_parser.hpp>
#include <ygm/utility.hpp>

#define DEFAULT_VORONOI_RANK 3
#define DEFAULT_NUM_HOPS 3

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
        << " -i <string>... 	 - File(s) containing data to build index from "
           "(required)\n"
        << " -q <string>...   - File(s) containing data to query index with "
           "(required)\n"
        << " -o <string>		- Output directory for query results "
           "(required)\n"
        << " -h            - print help and exit\n\n";
  }
}

void parse_cmd_line(int argc, char** argv, ygm::comm& comm, int& voronoi_rank,
                    int& num_hops, int& num_seeds, int& k,
                    std::vector<std::string>& index_filenames,
                    std::vector<std::string>& query_filenames,
                    std::string&              output_dir) {
  if (comm.rank0()) {
    std::cout << "CMD line:";
    for (int i = 0; i < argc; ++i) {
      std::cout << " " << argv[i];
    }
    std::cout << std::endl;
  }

  num_seeds               = -1;
  k                       = -1;
  bool found_voronoi_rank = false;
  bool found_num_hops     = false;

  int  c;
  bool inserting_index_filenames = false;
  bool inserting_query_filenames = false;
  bool prn_help                  = false;
  while (true) {
    while ((c = getopt(argc, argv, "+v:p:s:k:iqo:h ")) != -1) {
      inserting_index_filenames = false;
      inserting_query_filenames = false;
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
        case 'p':
          num_hops = atoi(optarg);
          break;
        case 'k':
          k = atoi(optarg);
          break;
        case 'i':
          inserting_index_filenames = true;
          break;
        case 'q':
          inserting_query_filenames = true;
          break;
        case 'o':
          output_dir = optarg;
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

    ++optind;
  }

  if (!found_voronoi_rank) {
    voronoi_rank = DEFAULT_VORONOI_RANK;
  }
  if (!found_num_hops) {
    num_hops = DEFAULT_NUM_HOPS;
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
  if (output_dir.size() < 1) {
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

template <typename String>
size_t levenshtein_distance(const String& s1, const String& s2) {
  const size_t m = s1.size();
  const size_t n = s2.size();

  if (m == 0) {
    return n;
  }
  if (n == 0) {
    return m;
  }

  // Row of matrix for dynamic programming approach
  std::vector<size_t> dist_row(m + 1);
  for (size_t i = 0; i < m + 1; ++i) {
    dist_row[i] = i;
  }

  for (size_t i = 1; i < n + 1; ++i) {
    size_t diag = i - 1;
    size_t next_diag;
    dist_row[0] = i;
    for (size_t j = 1; j < m + 1; ++j) {
      next_diag              = dist_row[j];
      bool substitution_cost = (s1[j - 1] != s2[i - 1]);

      dist_row[j] =
          std::min(1 + dist_row[j],
                   std::min(1 + dist_row[j - 1], substitution_cost + diag));
      diag = next_diag;
    }
  }

  return dist_row[m];
}

template <typename String>
float fuzzy_levenshtein(const String& s1, const String& s2) {
  float fuzz = (std::hash<String>()(s1) ^ std::hash<String>()(s2)) / 4.0;
  fuzz /= std::numeric_limits<size_t>::max();
  return levenshtein_distance(s1, s2) + fuzz;
}

template <typename String>
ygm::container::bag<String> read_ascii_lines(
    const std::vector<std::string>& filenames, ygm::comm& world) {
  ygm::container::bag<String> to_return(world);

  ygm::io::line_parser linep(world, filenames, false, true);

  linep.for_all(
      [&to_return](auto& ascii_line) { to_return.async_insert(ascii_line); });

  world.barrier();

  return to_return;
}

int main(int argc, char** argv) {
  ygm::comm world(&argc, &argv);

  using dist_t  = float;
  using index_t = std::size_t;
  using point_t = std::string;

  int                      voronoi_rank;
  int                      num_hops;
  int                      num_seeds;
  int                      k;
  std::vector<std::string> index_filenames;
  std::vector<std::string> query_filenames;
  std::string              output_dir;

  parse_cmd_line(argc, argv, world, voronoi_rank, num_hops, num_seeds, k,
                 index_filenames, query_filenames, output_dir);

  /*
for (int i = 1; i < argc; ++i) {
filenames.push_back(argv[i]);
}

int voronoi_rank = 3;
int num_seeds    = 4096;
  */

  auto ascii_lines = read_ascii_lines<std::string>(index_filenames, world);

  auto fuzzy_leven_space =
      saltatlas::dhnsw_detail::SpaceWrapper(fuzzy_levenshtein<std::string>);

  saltatlas::metric_hyperplane_partitioner<dist_t, index_t, point_t>
      fuzzy_partitioner(world, fuzzy_leven_space);

  ygm::container::pair_bag<index_t, point_t> bag_data(world);
  size_t                                     local_counter;
  ascii_lines.for_all(
      [&bag_data, &local_counter, world](const auto& ascii_line) {
        bag_data.async_insert(std::make_pair(
            (local_counter * world.size()) + world.rank(), ascii_line));
        ++local_counter;
      });

  world.barrier();

  saltatlas::dhnsw dist_index(voronoi_rank, num_seeds, &fuzzy_leven_space,
                              &world, fuzzy_partitioner);

  world.barrier();
  ygm::timer t{};

  dist_index.partition_data(bag_data, num_seeds);

  world.barrier();

  world.cout0("Distributing data to local HNSWs");
  bag_data.for_all([&dist_index, &world](const auto& ID_line) {
    dist_index.queue_data_point_insertion(ID_line.first, ID_line.second);
  });

  world.barrier();

  world.cout0("Initializing local HNSWs");
  dist_index.initialize_hnsw();

  world.barrier();
  world.cout0("Total build time: ", t.elapsed());

  world.cout0("Global HNSW size: ", dist_index.global_size());

  std::string test_string("example");
  world.cout0("Attempting query for '", test_string, "'");
  world.barrier();

  auto fuzzy_result_lambda =
      [](const point_t& query_string,
         const std::multimap<dist_t, std::pair<index_t, point_t>>&
              nearest_neighbors,
         auto dist_knn_index) {
        for (const auto& result_pair : nearest_neighbors) {
          dist_knn_index->comm().cout()
              << result_pair.second.second << " : d= " << result_pair.first
              << std::endl;
        }
      };
  if (world.rank0()) {
    dist_index.query_with_features(test_string, 5, 2, voronoi_rank, 1,
                                   fuzzy_result_lambda);
  }

  world.barrier();

  return 0;
}
