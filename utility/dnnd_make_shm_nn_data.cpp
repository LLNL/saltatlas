// Copyright 2020-2023 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <unistd.h>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <metall/container/string.hpp>
#include <metall/metall.hpp>
#include <ygm/comm.hpp>
#include <ygm/utility.hpp>

#include <saltatlas/dnnd/data_reader.hpp>
#include <saltatlas/dnnd/dhnsw_index_reader.hpp>
#include <saltatlas/dnnd/dnnd.hpp>
#include <saltatlas/dnnd/dnnd_pm.hpp>
#include <saltatlas/dnnd/utility.hpp>

using id_type              = uint32_t;
using feature_element_type = float;
using distance_type        = float;

using dnnd_pm_type =
    saltatlas::dnnd_pm<id_type, feature_element_type, distance_type>;
using pm_neighbor_type = typename dnnd_pm_type::neighbor_type;

template <typename T>
using matrix_type = metall::container::vector<
    metall::container::vector<T>,
    metall::manager::scoped_allocator_type<metall::container::vector<T>>>;
using point_store_type = matrix_type<feature_element_type>;
using knn_index_type   = matrix_type<id_type>;

int main(int argc, char** argv) {
  ygm::comm comm(&argc, &argv);

  std::string input_path;
  std::string out_path;
  bool        verbose{true};

  {
    int c;
    while ((c = getopt(argc, argv, "i:o:v")) != -1) {
      switch (c) {
        case 'i':
          input_path = optarg;
          break;
        case 'o':
          out_path = optarg;
          break;
        case 'v':
          verbose = false;
          break;
        default:
          comm.cerr0()
              << "Usage: " << argv[0]
              << " -i <input datastore path> -o <output datastore path> [-v]"
              << std::endl;
          MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
      }
    }
  }
  comm.cout0() << "Input datastore path: " << input_path << std::endl;
  comm.cout0() << "Output datastore path: " << out_path << std::endl;

  {
    std::unique_ptr<metall::manager> manager;
    static point_store_type*         main_point_store;
    static knn_index_type*           main_knn_index;
    if (comm.rank0()) {
      manager          = std::make_unique<metall::manager>(metall::create_only,
                                                  out_path.c_str());
      main_point_store = manager->construct<point_store_type>(
          metall::unique_instance)(manager->get_allocator());
      main_knn_index = manager->construct<knn_index_type>(
          metall::unique_instance)(manager->get_allocator());
    }
    comm.cf_barrier();

    dnnd_pm_type dnnd(dnnd_pm_type::open_read_only, input_path, comm, verbose);

    const auto& pstore = dnnd.get_point_store();
    const auto  max_id = comm.all_reduce_max(pstore.max_id());
    comm.cout0() << "Max ID: " << max_id << std::endl;
    if (comm.rank0()) {
      main_point_store->resize(max_id + 1);
      main_knn_index->resize(max_id + 1);
    }
    comm.cf_barrier();

    for (const auto& p : pstore) {
      const auto&                       id = p.first;
      std::vector<feature_element_type> feature(p.second.begin(),
                                                p.second.end());
      comm.async(
          0,
          [](auto, const auto& id, const auto& feature) {
            auto& row = main_point_store->at(id);
            row.insert(row.end(), feature.begin(), feature.end());
          },
          id, feature);
    }
    comm.barrier();
    comm.cout0() << "Constructed point store" << std::endl;

    const auto& knn_index = dnnd.get_knn_index();
    for (auto pitr = knn_index.points_begin(); pitr != knn_index.points_end();
         ++pitr) {
      const auto&          id = pitr->first;
      const auto&          nn = pitr->second;
      std::vector<id_type> knn(nn.size());
      std::transform(nn.cbegin(), nn.cend(), knn.begin(),
                     [](const auto& p) { return p.id; });
      comm.async(
          0,
          [](auto, const auto& id, const auto& knn) {
            auto& row = main_knn_index->at(id);
            row.insert(row.end(), knn.begin(), knn.end());
          },
          id, knn);
    }
    comm.barrier();

    if (comm.rank0()) {
      if (main_point_store->size() != max_id + 1) {
        comm.cerr0() << "Wrong #of points in point store: "
                     << main_point_store->size() << std::endl;
        MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
      }
      if (main_knn_index->size() != max_id + 1) {
        comm.cerr0() << "Wrong #of points in vertices in k-NN: "
                     << main_knn_index->size() << std::endl;
        MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
      }
    }
    comm.cout0() << "Constructed k-NN index" << std::endl;

    if (comm.rank0()) {
      manager->construct<metall::container::string>("distance-metric")(
          dnnd.get_distance_metric_name().c_str(), manager->get_allocator());
    }
    comm.cf_barrier();
  }  // Close Metall
  comm.cf_barrier();
  comm.cout0() << "Shared-memory NN data is ready to use." << std::endl;

  return EXIT_SUCCESS;
}