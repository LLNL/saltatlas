// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/*
 * hnswlib allows users to provide their own distance functions, but users must
 * do so through a DIST_TYPE(*) (const void *, const void *, const void*)
 * function pointer. Here, I provide a wrapper that allows users to define
 * distances in a slightly more C++ way. This method allows users to provide a
 * distance function that takes a pair of STL containers (among other
 * possibilities) without casting to const void *.
 */

#pragma once
#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

namespace saltatlas {
namespace utility {

template <typename DistType>
using hnswlib_func_type = DistType (*)(const void *, const void *,
                                       const void *);

/* This wrapper takes a more general distance function and places it in a
 * SpaceInterface object usable by hnswlib. It does this by using the third
 * "const void *" argument passed to distance functions in hnswlib to pass a
 * functor that wraps the desired distance function (needs to be a functor
 * because function pointers can't be casted to void *). The actual "distance
 * function" that gets passed to hnswlib is then a lambda that casts the data
 * points to the correct types and uses them as inputs to the functor (passed as
 * the extra argument, as detailed above).
 */
template <typename DistType, typename Point>
class SpaceWrapper : public hnswlib::SpaceInterface<DistType> {
 public:
  using wrapped_func_type = DistType (*)(Point &, Point &);

  SpaceWrapper(wrapped_func_type f) : m_dist_functor(f){};

  size_t get_data_size() { return sizeof(Point); }

  hnswlib_func_type<DistType> get_dist_func() { return m_arg_wrapper; }

  void *get_dist_func_param() { return &m_dist_functor; }

 private:
  struct dist_functor {
   public:
    dist_functor(wrapped_func_type f) : m_func(f){};

    DistType operator()(Point &x, Point &y) { return m_func(x, y); }

   private:
    wrapped_func_type m_func;
  };

  hnswlib_func_type<DistType> arg_wrapper() {
    auto l = [](const void *x_void_ptr, const void *y_void_ptr,
                const void *functor_ptr) {
      Point *x_ptr = (Point *)x_void_ptr;
      Point *y_ptr = (Point *)y_void_ptr;

      dist_functor dist_func = *((dist_functor *)functor_ptr);

      return dist_func(*x_ptr, *y_ptr);
    };

    return l;
  }

  dist_functor                m_dist_functor;
  hnswlib_func_type<DistType> m_arg_wrapper = arg_wrapper();
};

}  // namespace utility
}  // namespace saltatlas
