// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>

#include <ygm/comm.hpp>
#include <ygm/utility.hpp>

#include <saltatlas/dnnd/big_ann_bench_data_reader.hpp>
#include <saltatlas/dnnd/dnnd_pm.hpp>

using id_type         = uint32_t;
namespace data_reader = saltatlas::big_ann_bench_data_reader;

static constexpr std::size_t k_ygm_buff_size = 256 * 1024 * 1024;

struct option_t {
  int         index_k{0};
  double      r{0.8};
  double      delta{0.001};
  bool        exchange_reverse_neighbors{false};
  std::size_t batch_size{0};
  std::string distance_metric_name;
  std::string point_file_name;
  std::string datastore_path;
  bool        make_index_undirected{false};
  double      pruning_degree_multiplier{1.5};
  bool        remove_long_paths{false};
  int         query_k{4};
  std::string query_file_name;
  std::string ground_truth_file_name;
  std::string datastore_transfer_path;
  std::string out_prefix;
  bool        verbose{false};
};

bool parse_options(int argc, char **argv, option_t &option, bool &help);

template <typename cout_type>
void usage(std::string_view exe_name, cout_type &cout);

/// \brief Gather query results to the root rank.
template <typename neighbor_type, typename query_result_store_type>
inline std::vector<std::vector<neighbor_type>> gather_query_result(
    const query_result_store_type &local_result, ygm::comm &comm) {
  const std::size_t num_total_queries =
      comm.all_reduce_sum(local_result.size());

  static std::vector<std::vector<neighbor_type>> global_result;
  if (comm.rank0()) {
    global_result.resize(num_total_queries);
  }
  comm.cf_barrier();

  for (const auto &item : local_result) {
    const auto                 query_no = item.first;
    std::vector<neighbor_type> neighbors(item.second.begin(),
                                         item.second.end());
    if (neighbors.empty()) {
      std::cerr << query_no << "-th query result is empty (before sending)."
                << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    comm.async(
        0,
        [](const std::size_t                 query_no,
           const std::vector<neighbor_type> &neighbors) {
          global_result[query_no] = neighbors;
        },
        query_no, neighbors);
  }
  comm.barrier();

  // Sanity check
  if (comm.rank0()) {
    for (std::size_t i = 0; i < global_result.size(); ++i) {
      if (global_result[i].empty()) {
        std::cerr << i << "-th query result is empty (after gather)."
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
    }
  }

  return global_result;
}

template <typename neighbor_type>
inline void show_accuracy(
    const std::vector<std::vector<neighbor_type>> &ground_truth,
    const std::vector<std::vector<neighbor_type>> &test_result) {
  if (ground_truth.size() != test_result.size()) {
    std::cerr << "#of ground truth and test result data are different"
              << " ( " << ground_truth.size() << " , " << test_result.size()
              << " )" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  std::vector<double> accuracies;
  for (std::size_t i = 0; i < test_result.size(); ++i) {
    if (test_result[i].empty()) {
      std::cerr << "The " << i << "-th query result is empty" << std::endl;
      return;
    }

    if (ground_truth[i].empty()) {
      std::cerr << "The " << i << "-th ground truth is empty" << std::endl;
      return;
    }

    std::unordered_set<id_type> true_set;
    for (const auto &n : ground_truth[i]) true_set.insert(n.id);

    std::size_t num_corrects = 0;
    for (const auto &n : test_result[i]) {
      num_corrects += true_set.count(n.id);
    }

    accuracies.push_back((double)num_corrects / (double)test_result[i].size() *
                         100.0);
  }

  std::sort(accuracies.begin(), accuracies.end());

  std::cout << "Min accuracy\t" << accuracies.front() << std::endl;
  std::cout << "Mean accuracy\t"
            << std::accumulate(accuracies.begin(), accuracies.end(), 0.0) /
                   accuracies.size()
            << std::endl;
  std::cout << "Max accuracy\t" << accuracies.back() << std::endl;
}

template <typename neighbor_type>
inline void dump_query_result(
    const std::vector<std::vector<neighbor_type>> &result,
    const std::string &out_file_name, ygm::comm &comm) {
  if (comm.rank0() && !out_file_name.empty()) {
    comm.cout0() << "Dump result to files with prefix " << out_file_name
                 << std::endl;
    std::ofstream ofs_neighbors;
    std::ofstream ofs_distances;
    ofs_neighbors.open(out_file_name + "-neighbors.txt");
    ofs_distances.open(out_file_name + "-distances.txt");
    if (!ofs_distances.is_open() || !ofs_neighbors.is_open()) {
      comm.cerr0() << "Failed to create search result file(s)" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    for (const auto &neighbors : result) {
      for (const auto &n : neighbors) {
        ofs_neighbors << n.id << "\t";
        ofs_distances << n.distance << "\t";
      }
      ofs_neighbors << "\n";
      ofs_distances << "\n";
    }
  }
}

template <typename data_type, typename distance_type = float>
void run(const option_t &option, ygm::comm &comm) {
  using dnnd_pm_type  = saltatlas::dnnd_pm<id_type, data_type, distance_type>;
  using neighbor_type = typename dnnd_pm_type::neighbor_type;

  bool create_new = true;
  if (!option.datastore_transfer_path.empty() &&
      dnnd_pm_type::openable(option.datastore_transfer_path)) {
    if (dnnd_pm_type::copy(option.datastore_transfer_path,
                           option.datastore_path)) {
      create_new = false;
      comm.cout0() << "\nTransferred index from "
                   << option.datastore_transfer_path << " to "
                   << option.datastore_path << std::endl;
    } else {
      comm.cerr0() << "Failed to transfer index." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  auto dnnd = (create_new)
                  ? dnnd_pm_type(dnnd_pm_type::create, option.datastore_path,
                                 option.distance_metric_name, comm,
                                 std::random_device{}(), option.verbose)
                  : dnnd_pm_type(dnnd_pm_type::open_read_only,
                                 option.datastore_path, comm, option.verbose);

  if (create_new) {
    comm.cout0() << "\n<<Read Points>>" << std::endl;
    ygm::timer point_read_timer;
    data_reader::read_points(option.point_file_name,
                             dnnd.get_point_partitioner(), option.verbose,
                             dnnd.get_point_store(), comm);
    comm.cout0() << "\nReading points took (s)\t" << point_read_timer.elapsed()
                 << std::endl;

    comm.cout0() << "\n<<Index Construction>>" << std::endl;
    ygm::timer const_timer;
    dnnd.construct_index(option.index_k, option.r, option.delta,
                         option.exchange_reverse_neighbors, option.batch_size);
    comm.cout0() << "\nIndex construction took (s)\t" << const_timer.elapsed()
                 << std::endl;

    comm.cout0() << "\n<<Index Optimization>>" << std::endl;
    ygm::timer optimization_timer;
    dnnd.optimize_index(option.make_index_undirected,
                        option.pruning_degree_multiplier,
                        option.remove_long_paths);
    comm.cout0() << "\nIndex optimization took (s)\t"
                 << optimization_timer.elapsed() << std::endl;

    if (!option.datastore_transfer_path.empty()) {
      comm.cout0() << "\nTransferring index data store "
                   << option.datastore_path << " to "
                   << option.datastore_transfer_path << std::endl;
      if (!dnnd.snapshot(option.datastore_transfer_path)) {
        comm.cerr0() << "\nFailed to transfer index." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
    }
  }

  if (option.query_file_name.empty()) return;

  comm.cout0() << "<<Query>>" << std::endl;
  comm.cout0() << "Reading queries" << std::endl;
  typename dnnd_pm_type::query_point_store_type query_points;
  data_reader::read_queries<dnnd_pm_type>(option.query_file_name, query_points,
                                          option.verbose);
  comm.cf_barrier();

  comm.cout0() << "Executing queries" << std::endl;
  ygm::timer query_timer;
  const auto local_query_result =
      dnnd.query_batch(query_points, option.query_k, option.batch_size);
  comm.cf_barrier();
  comm.cout0() << "\nProcessing queries took (s)\t" << query_timer.elapsed()
               << std::endl;

  const auto all_query_result =
      gather_query_result<neighbor_type>(local_query_result, comm);
  if (!option.ground_truth_file_name.empty() && comm.rank0()) {
    std::vector<std::vector<neighbor_type>> gt;
    data_reader::read_ground_truths(option.ground_truth_file_name, gt,
                                    option.verbose);
    show_accuracy(gt, all_query_result);

    if (!option.out_prefix.empty()) {
      dump_query_result(all_query_result, option.out_prefix, comm);
    }
  }
}

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv, k_ygm_buff_size);

  option_t option;
  bool     help;

  if (!parse_options(argc, argv, option, help)) {
    comm.cerr0() << "Invalid option" << std::endl;
    usage(argv[0], comm.cerr0());
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  if (help) {
    usage(argv[0], comm.cout0());
    return 0;
  }

  const std::string suffix = option.point_file_name.substr(
      option.point_file_name.find_last_of('.') + 1);

  if (suffix == "u8bin") {
    run<uint8_t>(option, comm);
  } else if (suffix == "i8bin") {
    run<int8_t>(option, comm);
  } else if (suffix == "fbin") {
    static_assert(sizeof(float) == 4, "float is not 32 bits.");
    run<float>(option, comm);
  }

  return 0;
}

inline bool parse_options(int argc, char **argv, option_t &option, bool &help) {
  help = false;

  int n;
  while ((n = ::getopt(argc, argv, "b:d:ef:g:k:r:lm:n:o:p:q:uvx:z:h")) != -1) {
    switch (n) {
      case 'b':
        option.batch_size = std::stoul(optarg);
        break;

      case 'd':
        option.delta = std::stod(optarg);
        break;

      case 'e':
        option.exchange_reverse_neighbors = true;
        break;

      case 'f':
        option.distance_metric_name = optarg;
        break;

      case 'g':
        option.ground_truth_file_name = optarg;
        break;

      case 'k':
        option.index_k = std::stoi(optarg);
        break;

      case 'r':
        option.r = std::stod(optarg);
        break;

      case 'l':
        option.remove_long_paths = true;
        break;

      case 'm':
        option.pruning_degree_multiplier = std::stod(optarg);
        break;

      case 'n':
        option.query_k = std::stoi(optarg);
        break;

      case 'o':
        option.out_prefix = optarg;
        break;

      case 'p':
        option.point_file_name = optarg;
        break;

      case 'q':
        option.query_file_name = optarg;
        break;

      case 'u':
        option.make_index_undirected = true;
        break;

      case 'v':
        option.verbose = true;
        break;

      case 'x':
        option.datastore_transfer_path = optarg;
        break;

      case 'z':
        option.datastore_path = optarg;
        break;

      case 'h':
        help = true;
        return true;

      default:
        return false;
    }
  }

  return true;
}

template <typename cout_type>
void usage(std::string_view exe_name, cout_type &cout) {
  cout << "Usage: mpirun -n [#of processes] " << exe_name
       << " [options (see below)] [list of input point files (required)]"
       << std::endl;

  cout
      << "Options:"
      << "\n\t-z [string, required] Path to an already constructed or store a "
         "new index."
      << "\n\t-f [string, required] Distance metric name:"
      << "\n\t\t'l2' (L2), 'cosine' (cosine similarity), or 'jaccard'"
         "(Jaccard index)."
      << "\n\t-p [string, required] Path to an input point file."
      << "\n\t-k [int] Number of neighbors to have for each point in the index."
      << "\n\t-r [double] Sample rate parameter (ρ) in NN-Descent."
      << "\n\t-d [double] Precision parameter (δ) in NN-Descent."
      << "\n\t-e If specified, generate reverse neighbors globally during the "
         "index construction."
      << "\n"
      << "\n\t-u  If specified, make the index undirected before the query."
      << "\n\t-m [double] Pruning degree multiplier (m) in PyNNDescent."
      << "\n\t\tCut every points' neighbors more than 'k' x 'm' before the "
         "query."
      << "\n\t-l If specified, remove long paths before the query."
      << "\n"
      << "\n\t-q [string] Path to a query file."
      << "\n\t-n [int] Number of nearest neighbors to find for each query "
         "point."
      << "\n\t-g [string] Path to a query ground truth file."
      << "\n"
      << "\n\t-x [string] If specified, transfer index to this path at the end."
      << "\n\t-b [long int] Batch size for the index construction (0 is the "
         "full batch mode)."
      << "\n\t-v If specified, turn on the verbose mode."
      << "\n\t-h Show this menu." << std::endl;
}