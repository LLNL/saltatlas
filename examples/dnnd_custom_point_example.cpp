// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

// Usage:
//   cd saltatlas/build
//   mpirun -n 2 ./examples/dnnd_custom_point_example

#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <saltatlas/dnnd/dnnd.hpp>
#include <ygm/comm.hpp>

using id_type       = uint32_t;
using point_type    = std::unordered_map<int, std::vector<int>>;
using distance_type = uint32_t;
using dnnd_type     = saltatlas::dnnd<id_type, point_type, distance_type>;

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);

  int         index_k{2};
  int         query_k{4};
  double      r{0.8};
  double      delta{0.001};
  double      epsilon{0.1};
  double      mu{0.0};
  bool        exchange_reverse_neighbors{true};
  bool        make_index_undirected{true};
  double      pruning_degree_multiplier{0.0};  // No pruning
  std::size_t batch_size{1ULL << 31};
  bool        verbose{true};

  auto distance_func = [](const point_type &a,
                          const point_type &b) -> distance_type {
    auto sum0 = 0.0;
    auto sum1 = 0.0;
    for (const auto &[key, value] : a) {
      for (const auto &v : value) {
        sum0 += v;
      }
    }
    for (const auto &[key, value] : b) {
      for (const auto &v : value) {
        sum1 += v;
      }
    }
    return std::abs(sum0 - sum1);
  };

  dnnd_type dnnd(distance_func, comm, std::random_device{}(), verbose);
  comm.cf_barrier();

  comm.cout0() << "<<Init Points>>" << std::endl;
  {
    auto &point_store = dnnd.get_point_store();
    // Init points
    for (int i = 0; i < 10; i++) {
      const id_type id    = i * comm.size() + comm.rank();
      auto         &point = point_store[id];
      const auto    size  = std::rand() % 10 + 1;
      for (int j = 0; j < size; j++) {
        point[j].push_back(std::rand() % 100);
      }
    }
  }

  comm.cout0() << "<<Index Construction>>" << std::endl;
  dnnd.construct_index(index_k, r, delta, exchange_reverse_neighbors,
                       batch_size);

  comm.cout0() << "\n<<Optimizing the index for query>>" << std::endl;
  dnnd.optimize_index(make_index_undirected, pruning_degree_multiplier);

  comm.cout0() << "\n<<Query>>" << std::endl;
  std::vector<dnnd_type::point_type> queries;
  for (int i = 0; i < 10; i++) {
    dnnd_type::point_type query;
    const auto            size = std::rand() % 10 + 1;
    for (int j = 0; j < size; j++) {
      query[j].push_back(std::rand() % 100);
    }
    queries.push_back(query);
  }

  comm.cout0() << "Executing queries" << std::endl;
  const auto query_results =
      dnnd.query_batch(queries, query_k, epsilon, mu, batch_size);

  return 0;
}
