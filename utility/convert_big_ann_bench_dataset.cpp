#include <unistd.h>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <saltatlas/dnnd/detail/utilities/file.hpp>
#include <saltatlas/dnnd/detail/utilities/general.hpp>

namespace dtl = saltatlas::dndetail;

enum file_type {
  null,
  base,         // Input feature vectors
  query,        // Query
  ground_truth  // Ground truth
};

bool parse_options(int argc, char **argv, std::string &input_file_path,
                   std::string &out_file_prefix, file_type &type,
                   int &num_threads, bool &help);

template <typename cout_type>
void show_usage(std::string_view exe_name, cout_type &cout);

template <typename data_type, typename dump_type>
void convert_base_file(const std::string_view &input_file_path,
                       const std::string_view &out_file_prefix,
                       const int               num_parallels = 0);

template <typename data_type, typename dump_type>
void convert_query_file(std::string_view input_file_path,
                        std::string_view out_file_path);

template <typename id_type, typename distance_type>
void convert_ground_truth_file(std::string_view input_file_path,
                               std::string_view out_file_path);

int main(int argc, char **argv) {
  std::string input_file_path;
  std::string out_file_path;
  file_type   type;
  int         num_threads;
  bool        help;

  if (!parse_options(argc, argv, input_file_path, out_file_path, type,
                     num_threads, help)) {
    show_usage(argv[0], std::cerr);
    std::abort();
  } else if (help) {
    show_usage(argv[0], std::cout);
    return EXIT_SUCCESS;
  }

  std::cout << "Input file: " << input_file_path << std::endl;
  std::cout << "Output file: " << out_file_path << std::endl;

  const std::string suffix =
      input_file_path.substr(input_file_path.find_last_of('.') + 1);

  if (type == file_type::base) {
    std::cout << "Convert a base file" << std::endl;
    if (suffix == "u8bin") {
      convert_base_file<uint8_t, unsigned int>(input_file_path, out_file_path);
    } else if (suffix == "i8bin") {
      convert_base_file<int8_t, signed int>(input_file_path, out_file_path);
    } else if (suffix == "fbin") {
      static_assert(sizeof(float) == 4, "float is not 32 bits.");
      convert_base_file<float, float>(input_file_path, out_file_path);
    }
  } else if (type == file_type::query) {
    std::cout << "Convert a query file" << std::endl;
    if (suffix == "u8bin") {
      convert_query_file<uint8_t, unsigned int>(input_file_path, out_file_path);
    } else if (suffix == "i8bin") {
      convert_query_file<int8_t, signed int>(input_file_path, out_file_path);
    } else if (suffix == "fbin") {
      static_assert(sizeof(float) == 4, "float is not 32 bits.");
      convert_query_file<float, float>(input_file_path, out_file_path);
    }
  } else if (type == file_type::ground_truth) {
    std::cout << "Convert a ground truth file" << std::endl;
    if (suffix != "bin") {
      std::cerr << "Unexpected file suffix" << std::endl;
      std::abort();
    }
    convert_ground_truth_file<uint32_t, float>(input_file_path, out_file_path);
  } else {
    std::cerr << "File type was not specified properly." << std::endl;
    std::abort();
  }

  std::cout << "Finished conversion." << std::endl;

  return EXIT_SUCCESS;
}

bool parse_options(int argc, char **argv, std::string &input_file_path,
                   std::string &out_file_prefix, file_type &type,
                   int &num_threads, bool &help) {
  input_file_path.clear();
  out_file_prefix.clear();
  type                = file_type::null;
  num_threads         = 0;
  help                = false;
  bool type_specified = false;

  int n;
  while ((n = ::getopt(argc, argv, "i:o:t:bqgh")) != -1) {
    switch (n) {
      case 'i':
        input_file_path = optarg;
        break;

      case 'o':
        out_file_prefix = optarg;
        break;

      case 't':
        num_threads = std::stoul(optarg);
        break;

      case 'b':
        if (type_specified) return false;
        type           = file_type::base;
        type_specified = true;
        break;

      case 'q':
        if (type_specified) return false;
        type           = file_type::query;
        type_specified = true;
        break;

      case 'g':
        if (type_specified) return false;
        type           = file_type::ground_truth;
        type_specified = true;
        break;

      case 'h':
        help = true;
        return true;

      default:
        std::cerr << "Invalid option" << std::endl;
        std::abort();
    }
  }

  if (input_file_path.empty() || out_file_prefix.empty() ||
      type == file_type::null) {
    return false;
  }

  return true;
}

template <typename cout_type>
void show_usage(std::string_view exe_name, cout_type &cout) {
  cout << "Usage:\n"
       << exe_name
       << " -i /path/to/input -o /path/to/output [file type: -b, -q, or -g]"
       << "\nOptions:"
       << "\n-i [string]\tInput file path (required)."
       << "\n-o [string]\tOutput file path (required)."
       << "\n-t [int]\t#of threads when converting a base file. Use the "
          "default #of threads if not specified."
       << "\n-b\tIf specified, input file is the base file."
       << "\n-q\tIf specified, input file is the query file."
       << "\n-g\tIf specified, input file is the ground truth file."
       << "\n-h\tShow this help." << std::endl;
}

inline int get_num_threads() noexcept {
#ifdef _OPENMP
  return ::omp_get_num_threads();
#else
  return 1;
#endif
}

inline int get_thread_num() noexcept {
#ifdef _OPENMP
  return ::omp_get_thread_num();
#else
  return 0;
#endif
}

template <typename data_type, typename dump_data_type>
void convert_base_file(const std::string_view &input_file_path,
                       const std::string_view &out_file_prefix,
                       const int               num_parallels) {
  if (num_parallels > 0) {
#ifdef _OPENMP
    ::omp_set_num_threads(num_parallels);
#endif
  }

#ifdef _OPENMP
#pragma omp parallel default(none) \
    shared(std::cout, std::cerr, input_file_path, out_file_prefix)
#endif
  {
    std::ifstream ifs(input_file_path.data(), std::ios::binary);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open " << input_file_path << std::endl;
      std::abort();
    }

    uint32_t num_points     = 0;
    uint32_t num_dimensions = 0;
    if (!ifs.read(reinterpret_cast<char *>(&num_points), sizeof(num_points)) ||
        !ifs.read(reinterpret_cast<char *>(&num_dimensions),
                  sizeof(num_dimensions))) {
      std::cerr << "Failed to read from " << input_file_path << std::endl;
      std::abort();
    }

    if (get_thread_num() == 0) {
      std::cout << "#of point\t" << num_points << std::endl;
      std::cout << "#of dimension\t" << num_dimensions << std::endl;
      std::cout << "Use " << get_num_threads() << " threads" << std::endl;
    }

    const auto feature_size = num_dimensions * sizeof(data_type);
    const auto range =
        dtl::partial_range(num_points, get_thread_num(), get_num_threads());
    ifs.seekg(range.first * feature_size);
    const auto             num_reads = range.second - range.first;
    std::vector<data_type> feature(feature_size);

    const std::string out_file_path =
        (get_num_threads() == 1) ? std::string(out_file_prefix)
                                 : std::string(out_file_prefix) + "-" +
                                       std::to_string(get_thread_num());
    std::ofstream ofs(out_file_path + ".txt");
    if (!ofs.is_open()) {
      std::cerr << "Failed to create " << out_file_path << std::endl;
      std::abort();
    }

    // std::cout << get_thread_num() << " will read " << num_reads << std::endl;
    for (std::size_t i = 0; i < num_reads; ++i) {
      if (!dtl::read_by_chunk(ifs, reinterpret_cast<char *>(feature.data()),
                              feature_size)) {
        std::cerr << "Failed to read from " << input_file_path << std::endl;
        std::abort();
      }
      for (std::size_t k = 0; k < feature.size(); ++k) {
        if (k > 0) ofs << "\t";
        ofs << static_cast<dump_data_type>(feature[k]);
      }
      ofs << "\n";
    }
    ofs.close();
    if (!ofs) {
      std::cerr << "Failed to write to " << out_file_path << std::endl;
      std::abort();
    }
  }
}

template <typename data_type, typename dump_data_type>
void convert_query_file(std::string_view input_file_path,
                        std::string_view out_file_path) {
  std::ifstream ifs(input_file_path.data(), std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open " << input_file_path << std::endl;
    std::abort();
  }

  uint32_t num_queries    = 0;
  uint32_t num_dimensions = 0;
  if (!ifs.read(reinterpret_cast<char *>(&num_queries), sizeof(num_queries)) ||
      !ifs.read(reinterpret_cast<char *>(&num_dimensions),
                sizeof(num_dimensions))) {
    std::cerr << "Failed to read from " << input_file_path << std::endl;
    std::abort();
  }

  std::cout << "#of point\t" << num_queries << std::endl;
  std::cout << "#of dimension\t" << num_dimensions << std::endl;

  const auto             feature_size = num_dimensions * sizeof(data_type);
  std::vector<data_type> feature(feature_size);

  std::ofstream ofs(out_file_path.data() + std::string(".txt"));
  if (!ofs.is_open()) {
    std::cerr << "Failed to create " << out_file_path << std::endl;
    std::abort();
  }

  for (id_t id = 0; id < num_queries; ++id) {
    if (!dtl::read_by_chunk(ifs, reinterpret_cast<char *>(feature.data()),
                            feature_size)) {
      std::cerr << "Failed to read from " << input_file_path << std::endl;
      std::abort();
    }
    for (std::size_t k = 0; k < feature.size(); ++k) {
      if (k > 0) ofs << "\t";
      ofs << static_cast<dump_data_type>(feature[k]);
    }
    ofs << "\n";
  }

  ofs.close();
  if (!ofs) {
    std::cerr << "Failed to write to " << out_file_path << std::endl;
    std::abort();
  }
}

template <typename id_type, typename distance_type>
void convert_ground_truth_file(std::string_view input_file_path,
                               std::string_view out_file_path) {
  std::ifstream ifs(input_file_path.data(), std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open " << input_file_path << std::endl;
    std::abort();
  }

  uint32_t num_queries   = 0;
  uint32_t num_neighbors = 0;
  if (!ifs.read(reinterpret_cast<char *>(&num_queries), sizeof(num_queries)) ||
      !ifs.read(reinterpret_cast<char *>(&num_neighbors),
                sizeof(num_neighbors))) {
    std::cerr << "Failed to read from " << input_file_path << std::endl;
    std::abort();
  }

  std::cout << "#of point\t" << num_queries << std::endl;
  std::cout << "#of neighbors\t" << num_neighbors << std::endl;

  std::ofstream ofs(out_file_path.data() + std::string(".txt"));
  if (!ofs.is_open()) {
    std::cerr << "Failed to create " << out_file_path << std::endl;
    std::abort();
  }

  std::vector<id_type>       ids_buf(num_queries * num_neighbors);
  std::vector<distance_type> distances_buf(num_queries * num_neighbors);

  if (!dtl::read_by_chunk(ifs, reinterpret_cast<char *>(ids_buf.data()),
                          ids_buf.size() * sizeof(id_type)) ||
      !dtl::read_by_chunk(ifs, reinterpret_cast<char *>(distances_buf.data()),
                          distances_buf.size() * sizeof(distance_type))) {
    std::cerr << "Failed to read from " << input_file_path << std::endl;
    std::abort();
  }

  for (id_type i = 0; i < num_queries; ++i) {
    for (std::size_t k = 0; k < num_neighbors; ++k) {
      if (k > 0) ofs << "\t";
      ofs << ids_buf[i * num_neighbors + k];
    }
    ofs << "\n";
  }

  for (id_type i = 0; i < num_queries; ++i) {
    for (std::size_t k = 0; k < num_neighbors; ++k) {
      if (k > 0) ofs << "\t";
      ofs << distances_buf[i * num_neighbors + k];
    }
    ofs << "\n";
  }

  ofs.close();
  if (!ofs) {
    std::cerr << "Failed to write to " << out_file_path << std::endl;
    std::abort();
  }
}