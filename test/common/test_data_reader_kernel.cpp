// Copyright 2025 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <iostream>

#include <saltatlas/common/detail/data_reader_kernel.hpp>
#include <saltatlas/common/detail/utilities/float.hpp>

using namespace saltatlas;

void test_parse_feature_vector_float() {
  const auto kernel = [](const std::string& input) {
    const auto ret = saltatlas::detail::parse_feature_vector<float>(input);
    if (ret.size() != 3 || detail::nearly_equal(ret[0], 1.0f) == false ||
        detail::nearly_equal(ret[1], 2.0f) == false ||
        detail::nearly_equal(ret[2], 3.0f) == false) {
      std::cerr << "Failed parsing feature vector: " << input << std::endl;
      std::abort();
    }
  };

  kernel("1.0 2.0 3.0");
  kernel("1 2 3");
  kernel("1.0e0 2.0e0 3.0e0");
  kernel("1.0E0 2.0E0 3.0E0");
  kernel("1.0e+0 2.0e+0 3.0e+0");
  kernel("1.0E+0 2.0E+0 3.0E+0");
  kernel("1.0e-0 2.0e-0 3.0e-0");
  kernel("1.0E-0 2.0E-0 3.0E-0");

  // Leading, trailing, and multiple spaces
  kernel(" 1.0 2.0 3.0");
  kernel("1.0 2.0 3.0 ");
  kernel("1.0  2.0  3.0");
  kernel(" 1.0  2.0  3.0 ");
}

void test_parse_feature_vector_with_id_float() {
  const auto kernel = [](const std::string& input, const uint32_t expected_id) {
    const auto [id, vec] =
        saltatlas::detail::parse_feature_vector_with_id<uint32_t, float>(input);
    if (id != expected_id) {
      std::cerr << "Failed parsing feature vector with ID: " << input
                << std::endl;
      std::abort();
    }
    if (vec.size() != 3 || detail::nearly_equal(vec[0], 1.0f) == false ||
        detail::nearly_equal(vec[1], 2.0f) == false ||
        detail::nearly_equal(vec[2], 3.0f) == false) {
      std::cerr << "Failed parsing feature vector with ID: " << input
                << std::endl;
      std::abort();
    }
  };

  kernel("0 1.0 2.0 3.0", 0);
  kernel("1 1.0 2.0 3.0", 1);
  kernel("42 1.0 2.0 3.0", 42);

  // Leading, trailing, and multiple spaces
  kernel(" 0 1.0 2.0 3.0", 0);
  kernel("0 1.0 2.0 3.0 ", 0);
  kernel("0  1.0  2.0  3.0", 0);
  kernel(" 0  1.0  2.0  3.0 ", 0);
}

void test_parse_feature_vector_csv_float() {
  const auto kernel = [](const std::string& input) {
    const auto ret = saltatlas::detail::parse_feature_vector<float>(input, ',');
    if (ret.size() != 3 || detail::nearly_equal(ret[0], 1.0f) == false ||
        detail::nearly_equal(ret[1], 2.0f) == false ||
        detail::nearly_equal(ret[2], 3.0f) == false) {
      std::cerr << "Failed parsing feature vector: " << input << std::endl;
      std::abort();
    }
  };

  kernel("1.0,2.0,3.0");
  kernel("1,2,3");
  kernel("1.0e0,2.0e0,3.0e0");
  kernel("1.0E0,2.0E0,3.0E0");
  kernel("1.0e+0,2.0e+0,3.0e+0");
  kernel("1.0E+0,2.0E+0,3.0E+0");
  kernel("1.0e-0,2.0e-0,3.0e-0");
  kernel("1.0E-0,2.0E-0,3.0E-0");

  // Leading, trailing, and multiple spaces
  kernel(" 1.0,2.0,3.0");
  kernel("1.0,2.0,3.0 ");
  kernel("1.0, 2.0, 3.0");
  kernel(" 1.0, 2.0, 3.0 ");
}

void test_parse_feature_vector_with_id_csv_float() {
  const auto kernel = [](const std::string& input, const uint32_t expected_id) {
    const auto [id, vec] =
        saltatlas::detail::parse_feature_vector_with_id<uint32_t, float>(input,
                                                                         ',');
    if (id != expected_id) {
      std::cerr << "Failed parsing feature vector with ID: " << input
                << std::endl;
      std::abort();
    }
    if (vec.size() != 3 || detail::nearly_equal(vec[0], 1.0f) == false ||
        detail::nearly_equal(vec[1], 2.0f) == false ||
        detail::nearly_equal(vec[2], 3.0f) == false) {
      std::cerr << "Failed parsing feature vector with ID: " << input
                << std::endl;
      std::abort();
    }
  };

  kernel("0,1.0,2.0,3.0", 0);
  kernel("1,1.0,2.0,3.0", 1);
  kernel("42,1.0,2.0,3.0", 42);

  // Leading, trailing, and multiple spaces
  kernel(" 0,1.0,2.0,3.0", 0);
  kernel("0,1.0,2.0,3.0 ", 0);
  kernel("0, 1.0, 2.0, 3.0", 0);
  kernel(" 0, 1.0, 2.0, 3.0 ", 0);
}

// Make the same tests for uint8_t
void test_parse_feature_vector_uint8() {
  const auto kernel = [](const std::string& input) {
    const auto ret = saltatlas::detail::parse_feature_vector<uint8_t>(input);
    if (ret.size() != 3 || ret[0] != 1 || ret[1] != 2 || ret[2] != 3) {
      std::cerr << "Failed parsing feature vector: " << input << std::endl;
      std::abort();
    }
  };

  kernel("1 2 3");
  kernel(" 1 2 3");
  kernel("1 2 3 ");
  kernel("1  2  3");
  kernel(" 1  2  3 ");
}

void test_parse_feature_vector_with_id_uint8() {
  const auto kernel = [](const std::string& input, const uint32_t expected_id) {
    const auto [id, vec] =
        saltatlas::detail::parse_feature_vector_with_id<uint32_t, uint8_t>(
            input);
    if (id != expected_id) {
      std::cerr << "Failed parsing feature vector with ID: " << input
                << std::endl;
      std::abort();
    }
    if (vec.size() != 3 || vec[0] != 1 || vec[1] != 2 || vec[2] != 3) {
      std::cerr << "Failed parsing feature vector with ID: " << input
                << std::endl;
      std::abort();
    }
  };

  kernel("0 1 2 3", 0);
  kernel("1 1 2 3", 1);
  kernel("42 1 2 3", 42);

  // Leading, trailing, and multiple spaces
  kernel(" 0 1 2 3", 0);
  kernel("0 1 2 3 ", 0);
  kernel("0  1  2  3", 0);
  kernel(" 0  1  2  3 ", 0);
}

void test_parse_feature_vector_with_id_csv_uint8() {
  const auto kernel = [](const std::string& input, const uint32_t expected_id) {
    const auto [id, vec] =
        saltatlas::detail::parse_feature_vector_with_id<uint32_t, uint8_t>(
            input, ',');
    if (id != expected_id) {
      std::cerr << "Failed parsing feature vector with ID: " << input
                << std::endl;
      std::abort();
    }
    if (vec.size() != 3 || vec[0] != 1 || vec[1] != 2 || vec[2] != 3) {
      std::cerr << "Failed parsing feature vector with ID: " << input
                << std::endl;
      std::abort();
    }
  };

  kernel("0,1,2,3", 0);
  kernel("1,1,2,3", 1);
  kernel("42,1,2,3", 42);

  // Leading, trailing, and multiple spaces
  kernel(" 0,1,2,3", 0);
  kernel("0,1,2,3 ", 0);
  kernel("0, 1, 2, 3", 0);
  kernel(" 0, 1, 2, 3 ", 0);
}

void test_parse_feature_vector_csv_uint8() {
  const auto kernel = [](const std::string& input) {
    const auto ret =
        saltatlas::detail::parse_feature_vector<uint8_t>(input, ',');
    if (ret.size() != 3 || ret[0] != 1 || ret[1] != 2 || ret[2] != 3) {
      std::cerr << "Failed parsing feature vector: " << input << std::endl;
      std::abort();
    }
  };

  kernel("1,2,3");
  kernel(" 1,2,3");
  kernel("1,2,3 ");
  kernel("1, 2, 3");
  kernel(" 1, 2, 3 ");
}

int main(int argc, char** argv) {
  test_parse_feature_vector_float();
  test_parse_feature_vector_with_id_float();
  test_parse_feature_vector_csv_float();
  test_parse_feature_vector_with_id_csv_float();

  test_parse_feature_vector_uint8();
  test_parse_feature_vector_with_id_uint8();
  test_parse_feature_vector_csv_uint8();
  test_parse_feature_vector_with_id_csv_uint8();

  std::cout << "SUCCEEDED: " << argv[0] << std::endl;
  return 0;
}