// Copyright 2025 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

// Usage example:
// cd saltatlas/build
// mpirun -n 2 ./test/common/test_data_reader
// # Will work with an arbitrary number of ranks

#include <filesystem>
#include <iostream>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/common/data_reader.hpp>
#include <saltatlas/common/point_store.hpp>
#include <saltatlas/dnnd/feature_vector.hpp>

std::filesystem::path gen_file_path(const std::string& prefix,
                                    const int          file_index) {
  return std::filesystem::path(prefix + "_" + std::to_string(file_index) +
                               ".txt");
}

std::string serialize(const std::vector<float>& point,
                      const char                delimiter = ' ') {
  std::ostringstream oss;
  for (int i = 0; i < point.size(); ++i) {
    oss << point[i];
    if (i + 1 < point.size()) {
      oss << delimiter;
    }
  }
  return oss.str();
}

// Dump points into one or more files.
template <typename point_t>
std::vector<std::filesystem::path> create_test_files(
    const std::vector<point_t>&  points,
    const std::filesystem::path& path_prefix, const bool dump_id,
    const int num_splits, const char delimiter) {
  assert(num_splits > 0);

  const int   num_points      = points.size();
  const int   points_per_file = (num_points + num_splits - 1) / num_splits;
  std::size_t point_index     = 0;
  std::vector<std::filesystem::path> paths;
  std::size_t                        pid = 0;
  for (int fi = 0; fi < num_splits; ++fi) {
    paths.push_back(gen_file_path(path_prefix.string(), fi));
    std::ofstream ofs(paths.back());
    if (!ofs.is_open()) {
      std::cerr << "Failed to open " << paths.back() << std::endl;
      return {};
    }

    for (int pi = 0; pi < points_per_file && point_index < num_points;
         ++pi, ++point_index) {
      if (dump_id) {
        ofs << pid++ << delimiter;
      }
      if constexpr (std::is_same_v<typename point_t::value_type, char>) {
        for (int ci = 0; ci < points[point_index].size(); ++ci) {
          ofs << points[point_index][ci];
        }
        ofs << "\n";
      } else {
        ofs << serialize(points[point_index], delimiter) << "\n";
      }
    }
    ofs.close();
  }

  return paths;
}

template <typename point_t>
void run_test(const std::string&          format,
              const std::vector<point_t>& test_points,
              const std::string& file_prefix, const int num_files,
              ygm::comm& comm) {
  if (format == "wsv" || format == "wsv-id" || format == "csv" ||
      format == "csv-id" || format == "str" || format == "str-id") {
    // OK
  } else {
    comm.cerr0() << "Unsupported format: " << format << std::endl;
    MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
  }

  const char delimiter = (format.find("csv") != std::string::npos) ? ',' : ' ';
  const bool with_id   = (format.find("-id") != std::string::npos);

  const auto paths = create_test_files<point_t>(test_points, file_prefix,
                                                with_id, num_files, delimiter);
  comm.cf_barrier();

  // Read points from files
  using id_type = uint32_t;
  saltatlas::point_store<id_type, point_t> pstore;
  auto point_partitioner = [&comm](const id_type& id) {
    return id % comm.size();
  };
  saltatlas::read_points<id_type, point_t>(paths, format, false,
                                           point_partitioner, pstore, comm);

  // Check the read points
  const auto total_num_points = comm.all_reduce_sum(pstore.size());
  if (total_num_points != test_points.size()) {
    comm.cerr0() << "Number of read points mismatch: " << total_num_points
                 << " vs " << test_points.size() << std::endl;
    MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
  }

  for (const auto& [id, point] : pstore) {
    if (id >= test_points.size()) {
      comm.cerr() << "Invalid point ID: " << id << std::endl;
      MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
    }
    const auto& ref_point = test_points[id];
    if (point.size() != ref_point.size()) {
      comm.cerr() << "Point dimension mismatch: " << point.size() << " vs "
                  << ref_point.size() << std::endl;
      comm.cerr() << std::endl;
      MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
    }
    for (int i = 0; i < point.size(); ++i) {
      if (point[i] != ref_point[i]) {
        comm.cerr() << "Point value mismatch: " << point[i] << " vs "
                    << ref_point[i] << std::endl;
        MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
      }
    }
  }
  comm.cf_barrier();
}

int main(int argc, char** argv) {
  ygm::comm comm(&argc, &argv);

  if (comm.rank0()) {
    std::filesystem::create_directory("data_reader_test");
  }
  comm.cf_barrier();

  std::vector<std::vector<float>> test_float_points = {
      {61.58, 29.68, 20.43, 99.22, 21.81},
      {78.44, 54.43, 59.68, 65.80, 24.361},
      {12.58, 39.68, 90.43, 19.22, 81.81},
      {11.44, 14.43, 29.68, 35.80, 74.361},
      {51.58, 79.68, 50.43, 49.22, 41.81},
      {28.44, 24.43, 19.68, 15.80, 84.361},
      {41.58, 69.68, 40.43, 59.22, 31.81},
      {88.44, 94.43, 89.68, 95.80, 14.361},
      {31.58, 19.68, 30.43, 39.22, 71.81},
      {18.44, 34.43, 49.68, 25.80, 64.361}};

  // WSV without ID
  comm.cout0() << "Test wsv format" << std::endl;
  run_test("wsv", test_float_points, "data_reader_test/wsv", 1, comm);
  run_test("wsv", test_float_points, "data_reader_test/wsv", 2, comm);
  run_test("wsv", test_float_points, "data_reader_test/wsv", 3, comm);
  run_test("wsv", test_float_points, "data_reader_test/wsv", 4, comm);

  // WSV with ID
  comm.cout0() << "Test wsv-id format" << std::endl;
  run_test("wsv-id", test_float_points, "data_reader_test/wsv-id", 1, comm);
  run_test("wsv-id", test_float_points, "data_reader_test/wsv-id", 2, comm);
  run_test("wsv-id", test_float_points, "data_reader_test/wsv-id", 3, comm);
  run_test("wsv-id", test_float_points, "data_reader_test/wsv-id", 4, comm);

  // CSV without ID
  comm.cout0() << "Test csv format" << std::endl;
  run_test("csv", test_float_points, "data_reader_test/csv", 1, comm);
  run_test("csv", test_float_points, "data_reader_test/csv", 2, comm);
  run_test("csv", test_float_points, "data_reader_test/csv", 3, comm);
  run_test("csv", test_float_points, "data_reader_test/csv", 4, comm);

  // CSV with ID
  comm.cout0() << "Test csv-id format" << std::endl;
  run_test("csv-id", test_float_points, "data_reader_test/csv-id", 1, comm);
  run_test("csv-id", test_float_points, "data_reader_test/csv-id", 2, comm);
  run_test("csv-id", test_float_points, "data_reader_test/csv-id", 3, comm);
  run_test("csv-id", test_float_points, "data_reader_test/csv-id", 4, comm);

  std::vector<std::vector<char>> test_string_points(6);
  test_string_points[0].push_back('H');
  test_string_points[0].push_back('e');
  test_string_points[0].push_back('l');
  test_string_points[0].push_back('l');
  test_string_points[0].push_back('o');

  test_string_points[1].push_back('W');
  test_string_points[1].push_back('o');
  test_string_points[1].push_back('r');
  test_string_points[1].push_back('l');
  test_string_points[1].push_back('d');

  test_string_points[2].push_back('A');
  test_string_points[2].push_back('p');
  test_string_points[2].push_back('p');
  test_string_points[2].push_back('l');
  test_string_points[2].push_back('e');

  test_string_points[3].push_back('P');
  test_string_points[3].push_back('e');
  test_string_points[3].push_back('n');

  test_string_points[4].push_back('Z');

  test_string_points[5].push_back('1');
  test_string_points[5].push_back('2');
  test_string_points[5].push_back('3');
  test_string_points[5].push_back('4');

  // str without ID
  comm.cout0() << "Test str format" << std::endl;
  run_test("str", test_string_points, "data_reader_test/str", 1, comm);
  run_test("str", test_string_points, "data_reader_test/str", 2, comm);
  run_test("str", test_string_points, "data_reader_test/str", 3, comm);
  run_test("str", test_string_points, "data_reader_test/str", 4, comm);

  // str with ID
  comm.cout0() << "Test str-id format" << std::endl;
  run_test("str-id", test_string_points, "data_reader_test/str-id", 1, comm);
  run_test("str-id", test_string_points, "data_reader_test/str-id", 2, comm);
  run_test("str-id", test_string_points, "data_reader_test/str-id", 3, comm);
  run_test("str-id", test_string_points, "data_reader_test/str-id", 4, comm);

  comm.cout0() << "SUCCEEDED: " << argv[0] << std::endl;

  return 0;
}