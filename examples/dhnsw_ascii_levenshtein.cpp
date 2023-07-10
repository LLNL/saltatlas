
#include <unistd.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

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

namespace fs = std::filesystem;

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

  ygm::io::line_parser linep(world, filenames);

  linep.for_all(
      [&to_return](auto& ascii_line) { to_return.async_insert(ascii_line); });

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
        << " -o <string>		- Output directory for query results "
           "(required)\n"
        << " -h            - print help and exit\n\n";
  }
}

void parse_cmd_line(int argc, char** argv, ygm::comm& comm, int& voronoi_rank,
                    int& num_hops, int& num_seeds, int& k,
                    int&                      num_initial_queries,
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

  num_seeds                      = -1;
  k                              = -1;
  bool found_voronoi_rank        = false;
  bool found_num_hops            = false;
  bool found_initial_num_queries = false;

  int  c;
  bool inserting_index_filenames = false;
  bool inserting_query_filenames = false;
  bool prn_help                  = false;
  while (true) {
    while ((c = getopt(argc, argv, "+v:p:s:k:n:iqo:h ")) != -1) {
      inserting_index_filenames = false;
      inserting_query_filenames = false;
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

int main(int argc, char** argv) {
  ygm::comm world(&argc, &argv);

  using dist_t  = float;
  using index_t = std::size_t;
  using point_t = std::string;

  int                      voronoi_rank;
  int                      num_hops;
  int                      num_seeds;
  int                      k;
  int                      num_initial_queries;
  std::vector<std::string> index_filenames;
  std::vector<std::string> query_filenames;
  std::string              output_dir;

  parse_cmd_line(argc, argv, world, voronoi_rank, num_hops, num_seeds, k,
                 num_initial_queries, index_filenames, query_filenames,
                 output_dir);

  // Create directory for output
  if (output_dir.back() != '/') {
    output_dir.append("/");
  }
  fs::path output_dir_path(output_dir);
  fs::create_directories(output_dir_path);

  auto ascii_lines = read_ascii_lines<std::string>(index_filenames, world);

  auto fuzzy_leven_space =
      saltatlas::dhnsw_detail::SpaceWrapper(fuzzy_levenshtein<std::string>);

  saltatlas::metric_hyperplane_partitioner<dist_t, index_t, point_t>
      fuzzy_partitioner(world, fuzzy_leven_space);
  // saltatlas::voronoi_partitioner<dist_t, index_t, point_t> fuzzy_partitioner(
  // world, fuzzy_leven_space);

  ygm::container::bag<std::pair<index_t, point_t>> bag_data(world);
  size_t                                           local_counter;
  ascii_lines.for_all(
      [&bag_data, &local_counter, &world](const auto& ascii_line) {
        bag_data.async_insert(std::make_pair(
            (local_counter * world.size()) + world.rank(), ascii_line));
        ++local_counter;
      });

  world.barrier();

  world.cout0("===Building nearest neighbor index===");

  saltatlas::dhnsw dist_index(voronoi_rank, num_seeds, &fuzzy_leven_space,
                              &world, fuzzy_partitioner);

  world.barrier();
  ygm::timer t{};

  dist_index.partition_data(bag_data, num_seeds);

  world.barrier();

  world.cout0("Distributing data to local HNSWs");
  bag_data.for_all([&dist_index, &world](const auto& id, const auto& line) {
    dist_index.queue_data_point_insertion(id, line);
  });

  world.barrier();

  world.cout0("Initializing local HNSWs");
  dist_index.initialize_hnsw();

  world.barrier();
  world.cout0("Finished nearest neighbor index construction");
  world.cout0("Total build time: ", t.elapsed());
  world.cout0("Global HNSW size: ", dist_index.global_size());

  std::ofstream ofs(output_dir_path.string() + "nearest_neighbors" +
                    std::to_string(world.rank()));
  auto          p_ofs = world.make_ygm_ptr(ofs);

  auto query_lines = read_ascii_lines<std::string>(query_filenames, world);

  world.cout0("===Beginning queries===");
  world.barrier();
  t.reset();

  auto write_output_lambda =
      [](const point_t& query_string,
         const std::multimap<dist_t, std::pair<index_t, point_t>>&
              nearest_neighbors,
         auto ofs_ptr) {
        (*ofs_ptr) << query_string << " : [ ";
        for (const auto& result_pair : nearest_neighbors) {
          (*ofs_ptr) << "(" << result_pair.second.second << " : "
                     << result_pair.first << "),";
        }
        (*ofs_ptr) << " ]\n";
      };

  query_lines.for_all([&p_ofs, &dist_index, k, num_hops, num_initial_queries,
                       voronoi_rank,
                       write_output_lambda](const auto& query_str) {
    dist_index.query_with_features(query_str, k, num_hops, voronoi_rank,
                                   num_initial_queries, write_output_lambda,
                                   p_ofs);
  });

  world.barrier();

  world.cout0("Query time: ", t.elapsed());

  return 0;
}
