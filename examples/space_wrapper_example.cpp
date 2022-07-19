// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <stdlib.h>
#include <iostream>
#include <vector>

#include <hnswlib/hnswlib.h>
#include <saltatlas/dhnsw/detail/hnswlib_space_wrapper.hpp>

float my_l2_sqr(const std::vector<float> &x, const std::vector<float> &y) {
  if (x.size() != y.size()) {
    std::cerr << "Size mismatch for l2 distance" << std::endl;
    exit;
  }

  float dist_sqr{0.0};

  for (size_t i = 0; i < x.size(); ++i) {
    dist_sqr += (x[i] - y[i]) * (x[i] - y[i]);
  }

  return dist_sqr;
}

float my_projected_distance(std::vector<float> &x, std::vector<float> &y) {
  return abs(x[0] - y[0]);
}

int main(int argc, char **argv) {
  // Create HNSW with std::vector-based distance function
  {
    auto my_l2_space = saltatlas::dhnsw_detail::SpaceWrapper(my_l2_sqr);

    hnswlib::HierarchicalNSW<float> hnsw(&my_l2_space, 100, 16, 16, 1);

    // Define points to add to HNSW
    std::vector<float> p1{1.0, 0.0}, p2{0.0, 1.0}, p3{-1.0, 0.0}, p4{0.0, -1.0};
    std::vector<float> p5{5.0, 0.0}, p6{0.0, 5.0}, p7{-5.0, 0.0}, p8{0.0, -5.0};

    hnsw.addPoint(&p1, 1);
    hnsw.addPoint(&p2, 2);
    hnsw.addPoint(&p3, 3);
    hnsw.addPoint(&p4, 4);
    hnsw.addPoint(&p5, 5);
    hnsw.addPoint(&p6, 6);
    hnsw.addPoint(&p7, 7);
    hnsw.addPoint(&p8, 8);

    std::vector<float> q{0.0, 0.0};

    auto nn = hnsw.searchKnn(&q, 4);

    std::cout << "My l2 nearest neighbors: ";
    std::cout << nn.top().second;
    while (nn.size() > 1) {
      nn.pop();
      std::cout << ", " << nn.top().second;
    }
    std::cout << std::endl;
  }

  // HNSW with distances computed by projecting data onto first dimension. Using
  // std::vector allows us to have points in different numbers of dimensions.
  // Similar tricks may be helpful for more exotic spaces and distance functions
  // (think strings/text data)
  {
    auto my_projected_space =
        saltatlas::dhnsw_detail::SpaceWrapper(my_projected_distance);

    hnswlib::HierarchicalNSW<float> hnsw(&my_projected_space, 100, 16, 16, 1);

    // Define points to add to HNSW
    std::vector<float> p1{1.0, 0.0}, p2{0.0, 1.0}, p3{-1.0, 0.0}, p4{0.0, -1.0};
    std::vector<float> p5{5.0, 0.0, 0.0}, p6{0.0, 5.0, 0.0}, p7{-5.0, 0.0, 0.0},
        p8{0.0, -5.0, 0.0};

    hnsw.addPoint(&p1, 1);
    hnsw.addPoint(&p2, 2);
    hnsw.addPoint(&p3, 3);
    hnsw.addPoint(&p4, 4);
    hnsw.addPoint(&p5, 5);
    hnsw.addPoint(&p6, 6);
    hnsw.addPoint(&p7, 7);
    hnsw.addPoint(&p8, 8);

    std::vector<float> q{0.0, 0.0};

    auto nn = hnsw.searchKnn(&q, 4);

    std::cout << "My projected nearest neighbors: ";
    std::cout << nn.top().second;
    while (nn.size() > 1) {
      nn.pop();
      std::cout << ", " << nn.top().second;
    }
    std::cout << std::endl;
  }

  return 0;
}
