
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

    ASSERT_RELEASE(pt.size() == 128);

    to_return.async_insert(std::make_pair(id, pt));
  });

  world.barrier();

  return to_return;
}

ygm::container::pair_bag<index_t, point_t> read_query_points(
    const std::string& filename, ygm::comm& world) {
  ygm::container::pair_bag<index_t, point_t> to_return(world);

  if (world.rank0()) {
    size_t curr_line{0};

    std::ifstream ifs(filename);
    ifs.imbue(std::locale::classic());

    std::string line;
    while (std::getline(ifs, line)) {
      std::stringstream ss(line);

      point_t   pt;
      feature_t feature;
      while (ss >> feature) {
        pt.push_back(feature);
      }

      to_return.async_insert(std::make_pair(curr_line, pt));
      ++curr_line;
    }
  }

  world.barrier();

  return to_return;
}

ygm::container::map<index_t, std::vector<std::pair<index_t, dist_t>>>
read_ground_truth(const std::string& filename, ygm::comm& world) {
  ygm::container::map<index_t, std::vector<std::pair<index_t, dist_t>>>
      nn_dist_truth(world);

  if (world.rank0()) {
    size_t curr_offset{0};
    int    num_lines{0};

    // Count lines
    {
      std::ifstream ifs(filename);
      ifs.imbue(std::locale::classic());

      std::string line;

      while (std::getline(ifs, line)) {
        num_lines++;
      }
    }

    int num_ground_truth_entries = num_lines / 2;

    std::vector<std::vector<std::pair<index_t, dist_t>>> truth_vec(
        num_ground_truth_entries);

    // Read ground-truth data
    {
      std::ifstream ifs(filename);
      ifs.imbue(std::locale::classic());

      int curr_line{0};

      std::string line;
      while (std::getline(ifs, line)) {
        std::stringstream ss(line);

        if (curr_line < num_ground_truth_entries) {
          // Get neighbor IDs
          index_t ngbr;
          while (ss >> ngbr) {
            truth_vec[curr_line].push_back(std::make_pair(ngbr, -1.0));
          }
        } else {
          dist_t dist;
          int    curr_k{0};
          while (ss >> dist) {
            truth_vec[curr_line - num_ground_truth_entries][curr_k].second =
                dist;

            ++curr_k;
          }
        }
        ++curr_line;
      }
    }

    for (size_t i = 0; i < truth_vec.size(); ++i) {
      nn_dist_truth.async_insert(i + curr_offset, truth_vec[i]);
    }
  }

  world.barrier();

  // Sanity check on ground-truth
  nn_dist_truth.for_all([](const auto& id_nn_vec_pair) {
    const auto& [id, nn_vec] = id_nn_vec_pair;

    for (const auto& nn_dist : nn_vec) {
      ASSERT_RELEASE(nn_dist.second >= 0.0);
    }
  });

  return nn_dist_truth;
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

void parse_cmd_line(int argc, char** argv, ygm::comm& comm,
                    std::vector<int>& voronoi_rank_vec,
                    std::vector<int>& num_hops_vec, int& num_seeds,
                    std::vector<int>&         k_vec,
                    std::vector<int>&         num_initial_queries_vec,
                    std::vector<std::string>& index_filenames,
                    std::string&              query_filename,
                    std::string&              ground_truth_filename) {
  if (comm.rank0()) {
    std::cout << "CMD line:";
    for (int i = 0; i < argc; ++i) {
      std::cout << " " << argv[i];
    }
    std::cout << std::endl;
  }

  num_seeds                      = -1;
  bool found_voronoi_rank        = false;
  bool found_num_hops            = false;
  bool found_initial_num_queries = false;

  int  c;
  bool inserting_index_filenames     = false;
  bool inserting_voronoi_ranks       = false;
  bool inserting_num_hops            = false;
  bool inserting_k                   = false;
  bool inserting_num_initial_queries = false;
  bool prn_help                      = false;
  while (true) {
    while ((c = getopt(argc, argv, "+vps:kniq:g:h ")) != -1) {
      inserting_index_filenames     = false;
      inserting_voronoi_ranks       = false;
      inserting_num_hops            = false;
      inserting_k                   = false;
      inserting_num_initial_queries = false;
      switch (c) {
        case 'h':
          prn_help = true;
          break;
        case 'v':
          inserting_voronoi_ranks = true;
          found_voronoi_rank      = true;
          break;
        case 's':
          num_seeds = atoi(optarg);
          break;
        case 'p':
          inserting_num_hops = true;
          found_num_hops     = true;
          break;
        case 'k':
          inserting_k = true;
          break;
        case 'n':
          inserting_num_initial_queries = true;
          found_initial_num_queries     = true;
          break;
        case 'i':
          inserting_index_filenames = true;
          break;
        case 'q':
          query_filename = optarg;
          break;
        case 'g':
          ground_truth_filename = optarg;
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
    if (inserting_voronoi_ranks) {
      voronoi_rank_vec.push_back(atoi(argv[optind]));
    }
    if (inserting_num_hops) {
      num_hops_vec.push_back(atoi(argv[optind]));
    }
    if (inserting_k) {
      k_vec.push_back(atoi(argv[optind]));
    }
    if (inserting_num_initial_queries) {
      num_initial_queries_vec.push_back(atoi(argv[optind]));
    }

    ++optind;
  }

  if (!found_voronoi_rank) {
    comm.cout0("Using default voronoi rank: ", DEFAULT_VORONOI_RANK);
    voronoi_rank_vec.push_back(DEFAULT_VORONOI_RANK);
  }
  if (!found_num_hops) {
    comm.cout0("Using default number of hops: ", DEFAULT_NUM_HOPS);
    num_hops_vec.push_back(DEFAULT_NUM_HOPS);
  }
  if (!found_initial_num_queries) {
    comm.cout0("Using default number of initial queries: ",
               DEFAULT_NUM_INITIAL_QUERIES);
    num_initial_queries_vec.push_back(DEFAULT_NUM_INITIAL_QUERIES);
  }

  // Detect misconfigured options
  if (index_filenames.size() < 1) {
    comm.cout0("Must specify file to build index from");
    prn_help = true;
  }
  if (query_filename.size() < 1) {
    comm.cout0("Must specify file(s) to specifying queries");
    prn_help = true;
  }
  if (ground_truth_filename.size() < 1) {
    comm.cout0("Must specify file(s) containing ground truth");
    prn_help = true;
  }
  if (num_seeds < 1) {
    comm.cout0(
        "Must specify a positive number of seeds to use for building "
        "distributed index");
    prn_help = true;
  }
  if (k_vec.size() < 1) {
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

  std::vector<int>         voronoi_rank_vec;
  std::vector<int>         num_hops_vec;
  int                      num_seeds;
  std::vector<int>         k_vec;
  std::vector<int>         num_initial_queries_vec;
  std::vector<std::string> index_filenames;
  std::string              query_filename;
  std::string              ground_truth_filename;

  parse_cmd_line(argc, argv, world, voronoi_rank_vec, num_hops_vec, num_seeds,
                 k_vec, num_initial_queries_vec, index_filenames,
                 query_filename, ground_truth_filename);

  std::sort(voronoi_rank_vec.begin(), voronoi_rank_vec.end());
  std::sort(num_hops_vec.begin(), num_hops_vec.end());
  std::sort(k_vec.begin(), k_vec.end());
  std::sort(num_initial_queries_vec.begin(), num_initial_queries_vec.end());

  auto index_points = read_points(index_filenames, world);

  auto l2_space = saltatlas::dhnsw_detail::SpaceWrapper(l2_sqr);

  saltatlas::metric_hyperplane_partitioner<dist_t, index_t, point_t>
      l2_partitioner(world, l2_space);
  // saltatlas::voronoi_partitioner<dist_t, index_t, point_t> fuzzy_partitioner(
  // world, fuzzy_leven_space);

  world.cout0("===Building nearest neighbor index===");

  saltatlas::dhnsw dist_index(voronoi_rank_vec.back(), num_seeds, &l2_space,
                              &world, l2_partitioner);

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

  auto query_points = read_query_points(query_filename, world);
  auto ground_truth = read_ground_truth(ground_truth_filename, world);
  auto num_queries  = query_points.size();

  world.cout0("Query size: ", query_points.size(),
              "\nGround truth size: ", ground_truth.size());

  world.cout0("===Beginning queries===");
  static std::vector<
      std::pair<index_t, std::vector<std::pair<index_t, dist_t>>>>
                query_results;
  static size_t total_correct;
  static size_t total_query_results;
  static dist_t total_correct_dist;
  static dist_t total_result_dist;

  for (const auto k : k_vec) {
    for (const auto num_initial_queries : num_initial_queries_vec) {
      for (const auto num_hops : num_hops_vec) {
        for (const auto voronoi_rank : voronoi_rank_vec) {
          world.cout0("\nVoronoi rank: ", voronoi_rank, "\nHops: ", num_hops,
                      "\nInitial queries: ", num_initial_queries, "\nk: ", k);

          query_results.clear();
          total_correct       = 0;
          total_query_results = 0;
          total_correct_dist  = 0.0;
          total_result_dist   = 0.0;

          world.barrier();
          t.reset();

          auto store_results_lambda = [](const point_t& query_point,
                                         const std::multimap<dist_t, index_t>&
                                                 nearest_neighbors,
                                         auto    dist_knn_index,
                                         index_t query_ID) {
            std::pair<index_t, std::vector<std::pair<index_t, dist_t>>> results;
            results.first = query_ID;

            for (const auto& dist_ID_pair : nearest_neighbors) {
              results.second.push_back(
                  std::make_pair(dist_ID_pair.second, dist_ID_pair.first));
            }

            query_results.push_back(results);
          };

          query_points.for_all([&dist_index, k, num_hops, num_initial_queries,
                                voronoi_rank,
                                store_results_lambda](const auto& query_point) {
            dist_index.query(query_point.second, k, num_hops,
                             num_initial_queries, voronoi_rank,
                             store_results_lambda, query_point.first);
          });

          world.barrier();

          auto elapsed = t.elapsed();
          world.cout0("Query time: ", elapsed);
          world.cout0("\t", num_queries / elapsed, " queries/sec");

          world.cout0("Checking results");

          auto calculate_recall_lambda =
              [](auto& id_truth_vec_pair,
                 const std::vector<std::pair<index_t, dist_t>>& results) {
                const auto& [id, truth_vec] = id_truth_vec_pair;

                std::set<index_t> truth_set;

                for (int i = 0; i < results.size(); ++i) {
                  truth_set.insert(truth_vec[i].first);
                  total_correct_dist += std::sqrt(truth_vec[i].second);
                }

                for (const auto& query_result : results) {
                  total_correct += truth_set.count(query_result.first);
                  total_result_dist += std::sqrt(query_result.second);
                }

                total_query_results += results.size();
              };

          for (const auto& index_nn_vec : query_results) {
            const auto& [id, nn_vec] = index_nn_vec;

            ground_truth.async_visit(id, calculate_recall_lambda, nn_vec);
          }

          world.barrier();

          size_t global_correct = world.all_reduce_sum(total_correct);
          size_t global_nearest_ngbrs =
              world.all_reduce_sum(total_query_results);
          dist_t global_correct_dist = world.all_reduce_sum(total_correct_dist);
          dist_t global_result_dist  = world.all_reduce_sum(total_result_dist);

          /*
        world.cout0("Total Correct: ", global_correct,
                "\nTotal query results: ", global_nearest_ngbrs);
                                                          */
          world.cout0("Recall: ",
                      ((float)global_correct) / global_nearest_ngbrs);

          world.cout0(
              "Total correct dist: ", global_correct_dist,
              "\nTotal result dist: ", global_result_dist,
              "\nApprox Ratio: ", global_result_dist / global_correct_dist);
        }
      }
    }
  }

  return 0;
}
