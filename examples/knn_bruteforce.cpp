// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <math.h>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

float l_p_dist(std::vector<float> &a, std::vector<float> &b, float p) {
  if (a.size() != b.size()) {
    std::cerr << "Dimension mismatch" << std::endl;
    exit(1);
  }

  float to_return(0.0);
  for (int i = 0; i < a.size(); ++i) {
    to_return += pow(std::abs(a[i] - b[i]), p);
  }
  return pow(to_return, float(1) / p);
}

void read_data(const std::vector<std::string>                      fnames,
               std::vector<std::pair<size_t, std::vector<float>>> &data_vec) {
  // Read all sample features
  // (Using Geoff's example, hardcoded to 8 features)
  for (size_t i = 0; i < fnames.size(); ++i) {
    std::ifstream ifs(fnames[i].c_str());
    std::string   line;
    while (std::getline(ifs, line)) {
      std::stringstream  ssline(line);
      size_t             index;
      float              val;
      std::vector<float> values;
      ssline >> index;
      for (size_t i = 0; i < 8; ++i) {
        ssline >> val;
        values.push_back(val);
      }
      data_vec.push_back(std::make_pair(index, std::move(values)));
    }
  }
}

void check_dpockets(
    std::vector<std::pair<size_t, std::vector<float>>> &data_vec) {
  int num_correct{0};

#pragma omp parallel default(none) shared(num_correct, data_vec, std::cout)
  {
    int thread_num_correct{0};

#pragma omp for
    for (int i = 0; i < data_vec.size(); ++i) {
      std::priority_queue<std::pair<float, int>> ngbr_distances;
      // Calculate distances to all other points
      for (int j = 0; j < data_vec.size(); ++j) {
        float dist = l_p_dist(data_vec[i].second, data_vec[j].second, 2);

        if (ngbr_distances.size() < 10) {
          ngbr_distances.push(std::make_pair(dist, data_vec[j].first));
        } else if (ngbr_distances.top().first > dist) {
          ngbr_distances.pop();
          ngbr_distances.push(std::make_pair(dist, data_vec[j].first));
        }
      }

      // Check neighbors
      int correct_pocket = data_vec[i].first / 10;
      while (ngbr_distances.size() > 0) {
        auto ngbr        = ngbr_distances.top().second;
        int  ngbr_pocket = ngbr / 10;
        if (correct_pocket != ngbr_pocket) {
          // std::cout << "Point " << i << " has incorrect neighbor " << ngbr
          //<< std::endl;
        } else {
          ++thread_num_correct;
        }
        ngbr_distances.pop();
      }
    }

#pragma omp critical
    num_correct += thread_num_correct;
  }

  std::cout << "Found " << num_correct << " correct neighbors with brute-force"
            << std::endl;
}

int main(int argc, char **argv) {
  std::vector<std::string> input_fnames;
  for (int i = 1; i < argc; ++i) {
    input_fnames.push_back(argv[i]);
  }

  std::vector<std::pair<size_t, std::vector<float>>> data;

  read_data(input_fnames, data);

  std::cout << "Dataset contains " << data.size() << " data points"
            << std::endl;

  check_dpockets(data);
}
