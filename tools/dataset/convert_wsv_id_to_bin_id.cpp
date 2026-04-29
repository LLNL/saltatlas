// Copyright 2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

// Usage example:
// OMP_NUM_THREADS=8 ./convert_wsv_id_to_bin_id -o /path/to/bin-id/prefix \
// /path/to/wsv-id/file-0.txt /path/to/wsv-id/file-1.txt

#include <unistd.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <saltatlas/common/detail/data_reader_kernel.hpp>
#include <saltatlas/dnnd/detail/utilities/omp.hpp>
#include <saltatlas/neo_dnnd/mpi.hpp>

namespace {
using id_type = uint32_t;
#ifdef SALTATLAS_FEATURE_ELEMENT_TYPE
using fe_type = SALTATLAS_FEATURE_ELEMENT_TYPE;
#else
using fe_type = float;
#endif

constexpr std::size_t k_io_buffer_size = 256 * 1024 * 1024;

inline void convert_file(const std::string& input_file,
                         const std::string& output_file) {
  std::ifstream ifs(input_file);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open: " << input_file << std::endl;
    std::abort();
  }

  std::vector<char> in_buf(k_io_buffer_size);
  ifs.rdbuf()->pubsetbuf(in_buf.data(), in_buf.size());

  std::ofstream ofs(output_file, std::ios::binary | std::ios::trunc);
  if (!ofs.is_open()) {
    std::cerr << "Failed to create " << output_file << std::endl;
    std::abort();
  }

  std::vector<char> out_buf(k_io_buffer_size);
  ofs.rdbuf()->pubsetbuf(out_buf.data(), out_buf.size());

  std::string line;
  line.reserve(64 * 1024);

  uint64_t num_points = 0;
  uint64_t dims       = 0;

  if (!std::getline(ifs, line)) {
    ofs.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));
    ofs.write(reinterpret_cast<const char*>(&dims), sizeof(dims));
    return;
  }

  const auto first =
      saltatlas::detail::parse_feature_vector_with_id<id_type, fe_type>(line);
  id_type              id     = first.first;
  std::vector<fe_type> values = first.second;

  dims = static_cast<uint64_t>(values.size());
  ofs.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));
  ofs.write(reinterpret_cast<const char*>(&dims), sizeof(dims));

  std::vector<char> point_buf(sizeof(id_type) +
                              values.size() * sizeof(fe_type));
  std::memcpy(point_buf.data(), &id, sizeof(id_type));
  if (!values.empty()) {
    std::memcpy(point_buf.data() + sizeof(id_type), values.data(),
                values.size() * sizeof(fe_type));
  }
  ofs.write(point_buf.data(), point_buf.size());
  ++num_points;

  while (std::getline(ifs, line)) {
    const auto [id, values] =
        saltatlas::detail::parse_feature_vector_with_id<id_type, fe_type>(line);
    if (values.size() != static_cast<std::size_t>(dims)) {
      std::cerr << "Dim mismatch in " << input_file << std::endl;
      std::abort();
    }

    std::memcpy(point_buf.data(), &id, sizeof(id_type));
    if (dims > 0) {
      std::memcpy(point_buf.data() + sizeof(id_type), values.data(),
                  static_cast<std::size_t>(dims) * sizeof(fe_type));
    }
    ofs.write(point_buf.data(), point_buf.size());
    ++num_points;
  }

  if (!ifs.eof() && (ifs.bad() || ifs.fail())) {
    std::cerr << "Failed reading data from " << input_file << std::endl;
    std::abort();
  }

  ofs.seekp(0);
  ofs.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));
  ofs.write(reinterpret_cast<const char*>(&dims), sizeof(dims));
  ofs.close();

  if (!ofs) {
    std::cerr << "Failed to write to " << output_file << std::endl;
    std::abort();
  }
}

inline bool parse_options(int argc, char** argv,
                          std::vector<std::string>& inputs,
                          std::string&              out_file_prefix) {
  inputs.clear();
  out_file_prefix.clear();

  int n;
  while ((n = ::getopt(argc, argv, "o:")) != -1) {
    switch (n) {
      case 'o':
        out_file_prefix = optarg;
        break;
      default:
        return false;
    }
  }

  for (int index = optind; index < argc; index++) {
    inputs.emplace_back(argv[index]);
  }

  return !inputs.empty() && !out_file_prefix.empty();
}

inline void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " -o <output-prefix> <wsv-id-file> [wsv-id-file...]\n";
}
}  // namespace

int main(int argc, char** argv) {
  int provided;
  ::MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  {
    saltatlas::mpi::communicator comm;
    std::ios::sync_with_stdio(false);

    std::vector<std::string> inputs;
    std::string              out_file_prefix;
    if (!parse_options(argc, argv, inputs, out_file_prefix)) {
      print_usage(argv[0]);
      return 1;
    }

    for (std::size_t i = 0; i < inputs.size(); ++i) {
      if (i % comm.size() != comm.rank()) continue;

      const auto out_file_name =
          out_file_prefix + "-" + std::to_string(i) + ".bin";
      std::cout << "Convert " << inputs[i] << " to " << out_file_name
                << std::endl;
      convert_file(inputs[i], out_file_name);
    }
    comm.cout0() << "Finished the conversion." << std::endl;
  }
  ::MPI_Finalize();
  return 0;
}
