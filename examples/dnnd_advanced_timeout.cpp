// Copyright 2024–2025 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/// \brief A simple example of using DNND's advanced API
/// Usage:
///     cd build
///     mpirun -n 2 ./example/dnnd_advanced_timeout

#include <iostream>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/dnnd_adv.hpp>

using id_t       = uint32_t;
using dist_t     = double;
using point_type = saltatlas::pm_feature_vector<float>;

int main(int argc, char** argv) {
  ygm::comm comm(&argc, &argv);

  using dnnd_t = saltatlas::dnnd_adv<id_t, point_type, dist_t>;
  int    k     = 40;
  double rho   = 0.5;
  double delta = 0.001;
  {
    dnnd_t g(saltatlas::create_only, "/tmp/dnnd-datastore", comm,
             std::random_device{}());
    std::vector<std::filesystem::path> paths{
        "../examples/datasets/fashion-mnist_200.txt"};
    g.load_points(paths.begin(), paths.end(), "wsv");

    double time_limit_seconds = 1;  // very short time limit for demonstration
    comm.cout0() << "Creating a new datastore and building the index with a "
                 << time_limit_seconds << " second(s) timeout." << std::endl;
    const auto index_id =
        g.build(saltatlas::distance::id::l2, k, rho, delta, time_limit_seconds);
  }

  {
    dnnd_t g(saltatlas::open_only, "/tmp/dnnd-datastore", comm,
             std::random_device{}());

    comm.cout0()
        << "Re-opening the datastore and updating the index with no timeout."
        << std::endl;
    double time_limit_seconds = 0;  // no timeout
    g.update(g.get_index_ids().front(), saltatlas::distance::id::l2, k, rho,
             delta, time_limit_seconds);
  }

  return 0;
}
