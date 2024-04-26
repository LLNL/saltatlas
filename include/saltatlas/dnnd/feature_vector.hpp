// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#if __has_include(<metall/container/vector.hpp>)
#ifndef SALTATLAS_DNND_USE_METALL_CONTAINER
#define SALTATLAS_DNND_USE_METALL_CONTAINER 1
#endif
#endif

#if SALTATLAS_DNND_USE_METALL_CONTAINER
#include <metall/container/vector.hpp>
#include <ygm/detail/cereal_boost_container.hpp>
#else
#include <vector>
#endif

#include <memory>
#include "saltatlas/dnnd/detail/utilities/allocator.hpp"

namespace saltatlas {

namespace {
namespace container =
#if SALTATLAS_DNND_USE_METALL_CONTAINER
    metall::container;
#else
    std;
#endif
}  // namespace

/// \brief Feature vector type.
template <typename Element, typename Allocator = std::allocator<Element>>
using feature_vector = container::vector<Element, Allocator>;

#if SALTATLAS_DNND_USE_METALL_CONTAINER
/// \brief Feature vector type with persistent memory support.
template <typename Element,
          typename Allocator = metall::manager::fallback_allocator<Element>>
using pm_feature_vector = container::vector<Element, Allocator>;
#endif
}  // namespace saltatlas