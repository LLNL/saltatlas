
#include <cstdlib>
#include <vector>

#include <ygm/comm.hpp>
#include <ygm/io/csv_parser.hpp>
#include <ygm/io/line_parser.hpp>

#include <saltatlas/dhnsw/dhnsw.hpp>
#include "dnnd_example_common.hpp"

#include <hnswlib/space_l2.h>

using id_type      = int;
using feature_type = float;
using point_type   = std::vector<feature_type>;
using dist_type    = float;

struct option_t {
  int                                num_partitions;
  std::vector<std::filesystem::path> point_file_names;
  std::string                        point_file_format;
  std::filesystem::path              query_file_path;
  std::filesystem::path              ground_truth_file_path;
  std::filesystem::path              query_result_file_path;
  int                                query_k{0};
  bool                               verbose{false};
};

dist_type l2_sqr(const point_type& v1, const point_type& v2) {
  if (v1.size() != v2.size()) {
    std::cout << "Size mismatch: " << v1.size() << " != " << v2.size()
              << std::endl;
  }
  assert(v1.size() == v2.size());

  dist_type d = 0;
  for (std::size_t i = 0; i < v1.size(); ++i) {
    d += (v2[i] - v1[i]) * (v2[i] - v1[i]);
  }

  return d;
}

bool parse_options(int argc, char** argv, option_t& opt, bool& help) {
  opt.point_file_names.clear();
  opt.point_file_format.clear();
  opt.num_partitions = -1;
  help               = false;

  int n;
  while ((n = ::getopt(argc, argv, "n:p:q:k:g:o:vh")) != -1) {
    switch (n) {
      case 'n':
        opt.num_partitions = std::stoi(optarg);
        break;

      case 'p':
        opt.point_file_format = optarg;
        break;

      case 'q':
        opt.query_file_path = optarg;
        break;

      case 'k':
        opt.query_k = std::stoi(optarg);
        break;

      case 'g':
        opt.ground_truth_file_path = optarg;
        break;

      case 'o':
        opt.query_result_file_path = optarg;
        break;

      case 'v':
        opt.verbose = true;
        break;

      case 'h':
        help = true;
        return true;

      default:
        return false;
    }
  }

  for (int index = optind; index < argc; index++) {
    opt.point_file_names.emplace_back(argv[index]);
  }

  if (opt.point_file_format.empty() || opt.point_file_names.empty()) {
    return false;
  }

  return true;
}

/*
std::pair<std::vector<id_type>, std::vector<point_type>> read_points(
    const std::vector<std::string>& filenames,
    const std::string& point_file_format, ygm::comm& world) {
  std::vector<id_type>    ids;
  std::vector<point_type> points;

  if (point_file_format == "wsv-id") {
    ygm::io::line_parser linep(world, filenames);

    linep.for_all([&ids, &points](auto& line) {
      id_type    id;
      point_type p;

      std::stringstream ss(line);

      ss >> id;

      feature_type feature;
      while (ss >> feature) {
        p.push_back(feature);
      }

      ids.push_back(id);
      points.push_back(p);
    });
  } else if (point_file_format == "csv-id") {
    ygm::io::csv_parser csvp(world, filenames);

    csvp.for_all([&ids, &points](auto& line) {
      ids.push_back(line[0].as_integer());

      std::vector<feature_type> point;
      for (int i = 1; i < line.size(); ++i) {
        point.push_back(line[i].as_double());
      }

      points.push_back(point);
    });
  } else {
    world.cerr0("Unknown point file format");
    exit(0);
  }

  return std::make_pair(ids, points);
}
*/

int main(int argc, char** argv) {
  ygm::comm world(&argc, &argv);

  option_t opt;
  bool     help{false};
  if (!parse_options(argc, argv, opt, help)) {
    return 0;
  }
  if (help) {
    return 0;
  }

  saltatlas::dhnsw_params params;
  if (opt.num_partitions > 0) {
    params.num_partitions = opt.num_partitions;
  } else {
    params.num_partitions = world.size();
  }

  saltatlas::dhnsw<id_type, point_type, dist_type> my_dhnsw(l2_sqr, world,
                                                            params);

  my_dhnsw.load_points(opt.point_file_names.begin(), opt.point_file_names.end(),
                       opt.point_file_format);

  // auto [ids, points] =
  // read_points(opt.point_file_names, opt.point_file_format, world);

  // my_dhnsw.add_points(ids.begin(), ids.end(), points.begin(), points.end());

  my_dhnsw.build();

  if (!opt.query_file_path.empty()) {
    world.cout0() << "\n<<Query>>" << std::endl;
    std::vector<point_type> queries;
    saltatlas::read_query(opt.query_file_path, queries, world);

    world.cout0() << "Executing queries" << std::endl;
    ygm::timer step_timer;
    const auto query_results =
        my_dhnsw.query(queries.begin(), queries.end(), opt.query_k);
    world.cf_barrier();
    world.cout0() << "\nProcessing queries took (s)\t" << step_timer.elapsed()
                  << std::endl;

    if (!opt.ground_truth_file_path.empty()) {
      show_query_recall_score(query_results, opt.ground_truth_file_path, world);
      show_query_recall_score_with_only_distance(
          query_results, opt.ground_truth_file_path, world);
      show_query_recall_score_with_distance_ties(
          query_results, opt.ground_truth_file_path, world);
    }
  }
  return 0;
}
