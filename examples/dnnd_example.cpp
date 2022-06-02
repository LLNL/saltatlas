#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <ygm/comm.hpp>
#include <ygm/utility.hpp>

#include <saltatlas/dnnd/dnnd.hpp>

using id_type              = uint32_t;
using feature_element_type = float;
using distance_type        = float;
using dnnd_type = saltatlas::dnnd<id_type, feature_element_type, distance_type>;
using feature_vector_type = typename dnnd_type::feature_vector_type;
using neighbor_type       = typename dnnd_type::neighbor_type;

static constexpr std::size_t k_ygm_buff_size = 256 * 1024 * 1024;

void parse_options(int argc, char **argv, int &index_k, int &query_k, double &r,
                   double &delta, bool &exchange_reverse_neighbors,
                   std::size_t &batch_size, std::string &distance_metric_name,
                   std::vector<std::string> &point_file_names,
                   std::string              &query_file_name,
                   std::string &ground_truth_neighbor_ids_file_name,
                   std::string &point_file_format, std::string &out_file_prefix,
                   bool &verbose);

/// \brief Reads a file that contain queries.
/// Each line is the feature vector of a query point.
/// Can read the white space separated format.
void read_query(const std::string                 &query_file,
                dnnd_type::query_point_store_type &query_points);

/// \brief Read a file that contain a list of nearest neighbor IDs.
/// Each line is the nearest neighbor IDs a point.
/// Can read the white space separated format.
std::vector<std::vector<id_type>> read_neighbor_ids(
    const std::string &neighbors_file);

/// \brief Gather query result to the root process.
std::vector<std::vector<neighbor_type>> gather_query_result(
    const typename dnnd_type::query_result_store_type &local_result,
    ygm::comm                                         &comm);

/// \brief Calculate and show accuracy
void show_accuracy(const std::vector<std::vector<id_type>> &ground_truth_result,
                   const std::vector<std::vector<neighbor_type>> &test_result);

/// \brief The root process dumps query results.
void dump_query_result(
    const std::vector<std::vector<neighbor_type>> &global_result,
    const std::string &out_file_name, ygm::comm &comm);

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv, k_ygm_buff_size);
  {
    int                      index_k{0};
    int                      query_k{0};
    double                   r{0.8};
    double                   delta{0.001};
    bool                     exchange_reverse_neighbors{false};
    std::size_t              batch_size{0};
    std::string              distance_metric_name;
    std::vector<std::string> point_file_names;
    std::string              query_file_name;
    std::string              ground_truth_neighbor_ids_file_name;
    std::string              point_file_format;
    std::string              out_file_prefix;
    bool                     verbose{false};

    parse_options(argc, argv, index_k, query_k, r, delta,
                  exchange_reverse_neighbors, batch_size, distance_metric_name,
                  point_file_names, query_file_name,
                  ground_truth_neighbor_ids_file_name, point_file_format,
                  out_file_prefix, verbose);

    dnnd_type dnnd(distance_metric_name, point_file_names, point_file_format,
                   comm, verbose);
    comm.cf_barrier();

    comm.cout0() << "<<Index Construction>>" << std::endl;
    {
      ygm::timer step_timer;
      dnnd.construct_index(index_k, r, delta, exchange_reverse_neighbors,
                           batch_size);
      comm.cf_barrier();
      comm.cout0() << "Index construction took (s)\t" << step_timer.elapsed()
                   << std::endl;
    }

    if (!out_file_prefix.empty()) {
      dnnd.dump_index(out_file_prefix + "-index");
    }

    if (!query_file_name.empty()) {
      comm.cout0() << "\n<<Query>>" << std::endl;

      comm.cout0() << "Reading queries" << std::endl;
      dnnd_type::query_point_store_type query_points;
      read_query(query_file_name, query_points);
      comm.cf_barrier();

      comm.cout0() << "Executing queries" << std::endl;
      ygm::timer step_timer;
      const auto query_result =
          dnnd.query_batch(query_points, query_k, batch_size);
      comm.cf_barrier();
      comm.cout0() << "Processing queries took (s)\t" << step_timer.elapsed()
                   << std::endl;

      if (!ground_truth_neighbor_ids_file_name.empty() ||
          !out_file_prefix.empty()) {
        const auto all_query_result = gather_query_result(query_result, comm);
        if (!ground_truth_neighbor_ids_file_name.empty() && comm.rank0()) {
          comm.cout0() << "\nCalculate accuracy (%)" << std::endl;
          const auto ground_truth_neighbors =
              read_neighbor_ids(ground_truth_neighbor_ids_file_name);
          show_accuracy(ground_truth_neighbors, all_query_result);
        }

        if (!out_file_prefix.empty()) {
          comm.cout0() << "\nDump query results" << std::endl;
          dump_query_result(all_query_result, out_file_prefix + "-query", comm);
        }
      }
    }
  }
  return 0;
}

inline void parse_options(int argc, char **argv, int &index_k, int &query_k,
                          double &r, double &delta,
                          bool                     &exchange_reverse_neighbors,
                          std::size_t              &batch_size,
                          std::string              &distance_metric_name,
                          std::vector<std::string> &point_file_names,
                          std::string              &query_file_name,
                          std::string &ground_truth_neighbor_ids_file_name,
                          std::string &point_file_format,
                          std::string &out_file_prefix, bool &verbose) {
  distance_metric_name.clear();
  point_file_names.clear();
  point_file_format.clear();
  out_file_prefix.clear();
  verbose = false;

  int n;
  while ((n = ::getopt(argc, argv, "k:r:d:o:f:p:eb:vq:n:g:")) != -1) {
    switch (n) {
      case 'k':
        index_k = std::stoi(optarg);
        break;

      case 'r':
        r = std::stod(optarg);
        break;

      case 'd':
        delta = std::stod(optarg);
        break;

      case 'e':
        exchange_reverse_neighbors = true;
        break;

      case 'f':
        distance_metric_name = optarg;
        break;

      case 'o':
        out_file_prefix = optarg;
        break;

      case 'p':
        point_file_format = optarg;
        break;

      case 'b':
        batch_size = std::stoul(optarg);
        break;

      case 'q':
        query_file_name = optarg;
        break;

      case 'n':
        query_k = std::stoi(optarg);
        break;

      case 'g':
        ground_truth_neighbor_ids_file_name = optarg;
        break;

      case 'v':
        verbose = true;
        break;

      default:
        std::cerr << "Invalid option" << std::endl;
        std::abort();
    }
  }

  for (int index = optind; index < argc; index++) {
    point_file_names.emplace_back(argv[index]);
  }
}

inline void read_query(const std::string                 &query_file,
                       dnnd_type::query_point_store_type &query_points) {
  std::ifstream ifs(query_file);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open " << query_file << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  id_type     id = 0;
  std::string buf;
  while (std::getline(ifs, buf)) {
    std::stringstream    ss(buf);
    feature_element_type p;
    feature_vector_type  feature;
    while (ss >> p) {
      feature.push_back(p);
    }
    query_points.feature_vector(id) = feature;
    ++id;
  }
}

inline std::vector<std::vector<neighbor_type>> gather_query_result(
    const typename dnnd_type::query_result_store_type &local_result,
    ygm::comm                                         &comm) {
  const std::size_t num_total_queries =
      comm.all_reduce_sum(local_result.size());

  static std::vector<std::vector<neighbor_type>> global_result;
  if (comm.rank0()) {
    global_result.resize(num_total_queries);
  }
  comm.cf_barrier();

  for (const auto &item : local_result) {
    const auto &query_no  = item.first;
    const auto &neighbors = item.second;
    comm.async(
        0,
        [](const std::size_t                 query_no,
           const std::vector<neighbor_type> &neighbors) {
          global_result[query_no] = neighbors;
        },
        query_no, neighbors);
  }
  comm.barrier();

  return global_result;
}

inline std::vector<std::vector<id_type>> read_neighbor_ids(
    const std::string &neighbors_file) {
  std::ifstream ifs(neighbors_file);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open " << neighbors_file << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  std::vector<std::vector<id_type>> neighbor_ids;
  std::string                       buf;
  while (std::getline(ifs, buf)) {
    std::stringstream    ss(buf);
    id_type              id;
    std::vector<id_type> list;
    while (ss >> id) {
      list.push_back(id);
    }
    neighbor_ids.push_back(list);
  }

  return neighbor_ids;
}

inline void show_accuracy(
    const std::vector<std::vector<id_type>>       &ground_truth_result,
    const std::vector<std::vector<neighbor_type>> &test_result) {
  assert(ground_truth_result.size() == test_result.size());

  std::vector<double> accuracies;
  for (std::size_t i = 0; i < test_result.size(); ++i) {
    std::unordered_set<id_type> true_set;
    for (const auto &n : ground_truth_result[i]) true_set.insert(n);

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