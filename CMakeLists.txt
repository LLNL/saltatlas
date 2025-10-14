# Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
# saltatlas Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Works with 3.11 and tested through 3.15 (not tested yet)
cmake_minimum_required(VERSION 3.14)

# ---------------------------------------------------------------------------- #
# CMake policy
# ---------------------------------------------------------------------------- #
# When using the URL download method with the ExternalProject_Add() or
# FetchContent_Declare() commands, sets the timestamps of extracted contents
# to the time of extraction.
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
    cmake_policy(SET CMP0135 NEW)
endif ()

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.30")
    cmake_policy(SET CMP0167 NEW)
endif ()

# ---------------------------------------------------------------------------- #
# saltatlas configuration
# ---------------------------------------------------------------------------- #
project(
    saltatlas
    VERSION 0.1
    LANGUAGES C CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(SALTATLAS_MAIN_PROJECT ON)
endif ()

# Controls whether to set up install boilerplate for project and depencencies.
# Expects CMAKE_INSTALL_PREFIX to be set to a suitable directory.
option(JUST_INSTALL_SALTATLAS "Skip executables" OFF)
option(SALTATLAS_USE_HNSWLIB "Use Hnswlib" ON)
option(SALTATLAS_USE_METALL "Use Metall" ON)
option(SALTATLAS_DETERMINISM "Avoid shuffling for deterministic output" OFF)
# User options to specify Boost source or URL
set(BOOST_SOURCE_DIR CACHE STRING "Path to Boost libraries directory already uncompressed. Must be a version that supports CMake.")
set(BOOST_FETCH_URL CACHE STRING "URL or file path to an archived Boost source. Must be a version that supports CMake.")
set(HNSWLIB_FETCH_GIT
    "https://github.com/bwpriest/hnswlib.git"
    CACHE STRING "GIT to fetch hnswlib")
set(HNSWLIB_GIT_BRANCH
    "feature/cmake_facelift"
    CACHE STRING "Name of GIT branch for hnswlib")
set(YGM_FETCH_GIT
    "https://github.com/llnl/ygm.git"
    CACHE STRING "GIT to fetch ygm")
set(YGM_GIT_BRANCH
    "master"
    CACHE STRING "Name of GIT branch for ygm")
set(METALL_FETCH_GIT
    "https://github.com/llnl/metall.git"
    CACHE STRING "GIT to fetch Metall")
set(METALL_GIT_BRANCH
    "master"
    CACHE STRING "Name of GIT branch for Metall")
# Link Boost UUID with libatomic fails on LC machines
option(BOOST_UUID_LINK_LIBATOMIC "Link Boost UUID with libatomic" OFF)
option(SALTATLAS_BUILD_TEST "Build the test" OFF)

# Only do these if this is the main project, and not if it is included through
# add_subdirectory
if (SALTATLAS_MAIN_PROJECT)

    # Let's ensure -std=c++xx instead of -std=g++xx
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Let's nicely support folders in IDE's
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Testing only available if this is the main app Note this needs to be done
    # in the main CMakeLists since it calls enable_testing, which must be in the
    # main CMakeLists.
    include(CTest)
endif ()

# Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if (EXISTS "${LOC_PATH}")
    message(
        FATAL_ERROR
        "You cannot build in a source directory "
        "(or any directory with a CMakeLists.txt file). "
        "Please make a build subdirectory. "
        "Feel free to remove CMakeCache.txt and CMakeFiles."
    )
endif ()

# Check if the compiler is GCC
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.3.1")
        message(
            WARNING
            "Your GCC version may be too old. Please consider using GCC version 9.3.1 or higher."
        )
    endif ()
else ()
    message(
        WARNING
        "You are not using GCC. This project has only been tested with GCC. Please consider using GCC version 9.3.1 or higher."
    )
endif ()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

#
# MPI
#
find_package(MPI REQUIRED)
function(setup_mpi_target target_name)
    target_include_directories(${target_name} PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(${target_name} PRIVATE MPI::MPI_CXX)
endfunction()

#
# OpenMP
#
find_package(OpenMP)
function(setup_omp_target target_name)
    if (NOT OpenMP_CXX_FOUND)
        message(WARNING "OpenMP CXX not found!")
    else ()
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    endif ()
endfunction()

#
# Boost
#
include(fetch_boost)

# List of boost components this project depends on
set(BOOST_COMPS json unordered container static_string) # YGM requires these
if (SALTATLAS_USE_METALL)
    # Metall requires these
    # Extend BOOST_COMPS
    list(APPEND BOOST_COMPS interprocess property_tree uuid)
endif ()

# We do not check if Boost is already populated, since fetch_boost() does that.
if (BOOST_SOURCE_DIR)
    fetch_boost(SOURCE ${BOOST_SOURCE_DIR} COMPONENTS ${BOOST_COMPS})
elseif (BOOST_FETCH_URL)
    fetch_boost(URL ${BOOST_FETCH_URL} COMPONENTS ${BOOST_COMPS})
else ()
    # Use default Boost version if not specified by the user
    fetch_boost(URL "https://github.com/boostorg/boost/releases/download/boost-1.88.0/boost-1.88.0-cmake.tar.gz"
        COMPONENTS ${BOOST_COMPS})
endif ()

#
# hnswlib
#
if (SALTATLAS_USE_HNSWLIB)
    # Do not compile any hnswlib tests
    set(JUST_INSTALL_HNSWLIB ON)
    FetchContent_Declare(
        hnswlib
        GIT_REPOSITORY ${HNSWLIB_FETCH_GIT}
        GIT_TAG ${HNSWLIB_GIT_BRANCH}
    )
    FetchContent_MakeAvailable(hnswlib)
endif ()

#
# YGM
#
set(JUST_INSTALL_YGM TRUE)
FetchContent_Declare(
    ygm
    GIT_REPOSITORY ${YGM_FETCH_GIT}
    GIT_TAG ${YGM_GIT_BRANCH}
)
FetchContent_MakeAvailable(ygm)

#
# SALTATLAS INTERFACE
#
add_library(saltatlas INTERFACE)
add_library(saltatlas::saltatlas ALIAS saltatlas)
target_compile_features(saltatlas INTERFACE cxx_std_20)
target_include_directories(
    saltatlas INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(
    saltatlas INTERFACE ${SALTATLAS_CEREAL_TARGET} ygm::ygm ${BOOST_LIBS}
)
if (SALTATLAS_USE_HNSWLIB)
    target_link_libraries(saltatlas INTERFACE hnswlib::hnswlib)
endif ()
if (SALTATLAS_DETERMINISM)
    message(STATUS ${PROJECT_NAME} " setting determinism")
    target_compile_definitions(saltatlas INTERFACE SALTATLAS_DETERMINISM)
endif ()

#
# Metall
#
if (SALTATLAS_USE_METALL)
    set(JUST_INSTALL_METALL_HEADER ON)
    FetchContent_Declare(
        Metall
        GIT_REPOSITORY ${METALL_FETCH_GIT}
        GIT_TAG ${METALL_GIT_BRANCH}
    )
    FetchContent_MakeAvailable(Metall)
    find_package(Threads REQUIRED)

    target_link_libraries(saltatlas INTERFACE Metall::Metall)
    target_link_libraries(saltatlas INTERFACE Threads::Threads)
endif ()

function(setup_metall_target target_name)
    if (SALTATLAS_USE_METALL)
        target_link_libraries(${target_name} PRIVATE Metall::Metall)
        target_link_libraries(${target_name} PRIVATE Threads::Threads)
    endif ()
endfunction()

if (JUST_INSTALL_SALTATLAS)
    return()
endif ()

if (SALTATLAS_MAIN_PROJECT)
    add_subdirectory(examples)
    add_subdirectory(tools)
    if (SALTATLAS_BUILD_TEST)
        enable_testing()
        add_subdirectory(test)
    endif ()
endif ()
