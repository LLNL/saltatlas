# Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
# saltatlas Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Works with 3.11 and tested through 3.15 (not tested yet)
cmake_minimum_required(VERSION 3.14)

# ---------------------------------------------------------------------------- #
# CMake policy
# ---------------------------------------------------------------------------- #
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
option(SALTATLAS_USE_METALL "Use Metall" OFF)
option(SALTATLAS_DETERMINISM "Avoid shuffling for deterministic output" OFF)

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

include(FetchContent)

#
# OpenMP
#
find_package(OpenMP)

#
# Boost
#
set(FETCHCONTENT_QUIET FALSE)
if (SALTATLAS_USE_METALL)
    FetchContent_Declare(
        Boost
        URL https://github.com/boostorg/boost/releases/download/boost-1.87.0/boost-1.87.0-cmake.tar.gz
    )
    set(BOOST_INCLUDE_LIBRARIES
        json
        unordered
        interprocess
        container
        property_tree
        uuid
        graph
    )
    FetchContent_MakeAvailable(Boost)
    foreach (lib ${BOOST_INCLUDE_LIBRARIES})
        list(APPEND BOOST_FETCHED_COMPONENTS "Boost::${lib}")
    endforeach ()
else ()
    FetchContent_Declare(
        Boost
        URL https://github.com/boostorg/boost/releases/download/boost-1.87.0/boost-1.87.0-cmake.tar.gz
    )
    set(BOOST_INCLUDE_LIBRARIES
        unordered
        container
    )
    FetchContent_MakeAvailable(Boost)
    foreach (lib ${BOOST_INCLUDE_LIBRARIES})
        list(APPEND BOOST_FETCHED_COMPONENTS "Boost::${lib}")
    endforeach ()
endif ()


#
# hnswlib
#
# Do not compile any hnswlib tests
set(JUST_INSTALL_HNSWLIB ON)
FetchContent_Declare(
    hnswlib
    GIT_REPOSITORY https://github.com/bwpriest/hnswlib.git
    GIT_TAG feature/cmake_facelift
)
FetchContent_MakeAvailable(hnswlib)

#
# Cereal
#
set(JUST_INSTALL_CEREAL TRUE)
FetchContent_Declare(
    cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal.git
    GIT_TAG v1.3.2
)
FetchContent_MakeAvailable(cereal)

#
# YGM
#
set(JUST_INSTALL_YGM TRUE)
FetchContent_Declare(
    ygm
    GIT_REPOSITORY https://github.com/LLNL/ygm.git
    GIT_TAG v0.8-dev
)
FetchContent_MakeAvailable(ygm)

add_library(saltatlas INTERFACE)
add_library(saltatlas::saltatlas ALIAS saltatlas)
target_compile_features(saltatlas INTERFACE cxx_std_20)
target_include_directories(
    saltatlas INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:include>
)
target_link_libraries(
    saltatlas INTERFACE ${SALTATLAS_CEREAL_TARGET} ygm::ygm hnswlib::hnswlib
)
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
        GIT_REPOSITORY https://github.com/LLNL/metall.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(Metall)

    find_package(Threads REQUIRED)
    target_link_libraries(saltatlas INTERFACE Threads::Threads Metall::Metall)

    target_link_libraries(saltatlas INTERFACE ${BOOST_FETCHED_COMPONENTS})
endif ()

if (JUST_INSTALL_SALTATLAS)
    return()
endif ()

if (SALTATLAS_MAIN_PROJECT)
    add_subdirectory(examples)
    add_subdirectory(utility)
    add_subdirectory(test)
endif ()
