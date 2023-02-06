# Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
# saltatlas Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Works with 3.11 and tested through 3.15 (not tested yet)
cmake_minimum_required(VERSION 3.14)

project(
    saltatlas
    VERSION 0.1
    LANGUAGES C CXX
)

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(SALTATLAS_MAIN_PROJECT ON)
endif ()

# Controls whether to set up install boilerplate for project and depencencies.
# Expects CMAKE_INSTALL_PREFIX to be set to a suitable directory.
option(SALTATLAS_INSTALL "Generate the install target"
       ${SALTATLAS_MAIN_PROJECT}
)

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

include(FetchContent)

option(SALTATLAS_USE_METALL "Use Metall" OFF)

#
# Boost
#
# Disable the boost-cmake feature (BoostConfig.cmake or boost-config.cmake)
# because there is a tricky behavior/issue. We use FetchContent to download
# Boost and find_package to find the location of Boost. As find_package natively
# supports Boost, we do not need to use the boost-cmake feature.
set(Boost_NO_BOOST_CMAKE ON)

find_package(Boost 1.64 QUIET)
if (NOT Boost_FOUND)
    FetchContent_Declare(
        Boost
        URL https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.bz2
    )
    FetchContent_GetProperties(Boost)
    if (Boost_POPULATED)
        message(STATUS ${PROJECT_NAME}
                       " found already populated Boost dependency: "
                       ${boost_SOURCE_DIR}
        )
    else ()
        if (SALTATLAS_USE_METALL) # Only Metall requires Boost
            FetchContent_Populate(Boost)
            message(STATUS ${PROJECT_NAME} " cloned Boost dependency: "
                           ${boost_SOURCE_DIR}
            )
        endif ()
    endif ()
    set(BOOST_ROOT ${boost_SOURCE_DIR})
    find_package(Boost 1.64)
else ()
    message(STATUS ${PROJECT_NAME} " found Boost dependency: "
                   ${Boost_INCLUDE_DIR}
    )
endif ()

#
# hnswlib
#
find_package(hnswlib CONFIG)
if (NOT hnswlib_FOUND)
    FetchContent_Declare(
        hnswlib
        GIT_REPOSITORY https://github.com/bwpriest/hnswlib
        GIT_TAG feature/cmake_facelift
    )
    FetchContent_GetProperties(hnswlib)
    if (hnswlib_POPULATED)
        message(STATUS ${PROJECT_NAME}
                       " found already populated hnswlib dependency: "
                       ${hnswlib_SOURCE_DIR}
        )
    else ()
        # Do not compile any hnswlib tests
        set(JUST_INSTALL_HNSWLIB ON)
        # Install hnswlib at ${CMAKE_INSTALL_PREFIX}
        set(HNSWLIB_INSTALL ${SALTATLAS_INSTALL})
        # Populate hnswlib
        FetchContent_Populate(hnswlib)
        # Include hnswlib root cmake boilerplate
        add_subdirectory(${hnswlib_SOURCE_DIR} ${hnswlib_BINARY_DIR})
        message(STATUS ${PROJECT_NAME} " cloned hnswlib dependency: "
                       ${hnswlib_SOURCE_DIR}
        )
    endif ()
else ()
    message(STATUS ${PROJECT_NAME} " found hnswlib dependency: " ${hnswlib_DIR})
    if (${hnswlib_SOURCE_DIR} MATCHES "*_deps*")
        add_subdirectory(${hnswlib_SOURCE_DIR} ${hnswlib_BINARY_DIR})
    endif ()
endif ()

#
# Cereal
#
set(SALTATLAS_CEREAL_VERSION 1.3.0)
# Will do something like this upon new release and fix to spack distribution.
# find_package(cereal ${SALTATLAS_CEREAL_VERSION} CONFIG QUIET)
find_package(cereal CONFIG QUIET)
set(SALTATLAS_CEREAL_TARGET "cereal::cereal")
if (NOT cereal_FOUND)
    # Currently cereal version 1.3.0 has an outdated CMakeLists.txt, so we need
    # to use this commit for now.
    FetchContent_Declare(
        cereal
        GIT_REPOSITORY https://github.com/USCiLab/cereal
        GIT_TAG af0700efb25e7dc7af637b9e6f970dbb94813bff
    )
    FetchContent_GetProperties(cereal)
    if (cereal_POPULATED)
        message(STATUS ${PROJECT_NAME}
                       " found already populated cereal dependency: "
                       ${cereal_SOURCE_DIR}
        )
    else ()
        # Do not compile any cereal tests
        set(JUST_INSTALL_CEREAL ON)
        # Install cereal at ${CMAKE_INSTALL_PREFIX}
        set(CEREAL_INSTALL ${SALTATLAS_INSTALL})
        # Populate cereal
        FetchContent_Populate(cereal)
        # Include cereal root cmake boilerplate
        add_subdirectory(${cereal_SOURCE_DIR} ${cereal_BINARY_DIR})
        message(STATUS ${PROJECT_NAME} " cloned cereal dependency: "
                       ${cereal_SOURCE_DIR}
        )
    endif ()
else ()
    message(STATUS ${PROJECT_NAME} " found cereal dependency: " ${cereal_DIR})
    if (${cereal_SOURCE_DIR} MATCHES "*_deps*")
        add_subdirectory(${cereal_SOURCE_DIR} ${cereal_BINARY_DIR})
    endif ()
    # cereal installed with spack creates library target "cereal", whereas
    # installing from source creates target "cereal::cereal". This is the only
    # simple way I could figure out how to differentiate the two, but this will
    # cause problems if a spack instance installs headers to a path that does
    # not include the substring "spack".
    if (${cereal_DIR} MATCHES ".*spack.*")
        set(SALTATLAS_CEREAL_TARGET "cereal")
    else ()

    endif ()
endif ()

#
# YGM
#
set(SALTATLAS_YGM_VERSION 0.5)
find_package(ygm ${SALTATLAS_YGM_VERSION} QUIET CONFIG)
if (NOT ygm_FOUND)
    FetchContent_Declare(
        ygm
        GIT_REPOSITORY https://github.com/bwpriest/ygm
        GIT_TAG feature/bag_of_pairs # v${SALTATLAS_YGM_VERSION}
    )
    FetchContent_GetProperties(ygm)
    if (ygm_POPULATED)
        message(STATUS ${PROJECT_NAME}
                       " found already populated ygm dependency: "
                       ${ygm_SOURCE_DIR}
        )
    else ()
        # Do not compile any ygm tests
        set(JUST_INSTALL_YGM ON)
        # Install ygm at ${CMAKE_INSTALL_PREFIX}
        set(YGM_INSTALL ${SALTATLAS_INSTALL})
        # Populate ygm
        FetchContent_Populate(ygm)
        # Include ygm root cmake boilerplate
        add_subdirectory(${ygm_SOURCE_DIR} ${ygm_BINARY_DIR})
        message(STATUS ${PROJECT_NAME} " cloned ygm dependency "
                       ${ygm_SOURCE_DIR}
        )
    endif ()
else ()
    message(STATUS ${PROJECT_NAME} " found ygm dependency " ${ygm_DIR})
    if (${ygm_SOURCE_DIR} MATCHES "*_deps*")
        add_subdirectory(${ygm_SOURCE_DIR} ${ygm_BINARY_DIR})
    endif ()
endif ()

add_library(saltatlas INTERFACE)
add_library(saltatlas::saltatlas ALIAS saltatlas)
target_compile_features(saltatlas INTERFACE cxx_std_17)
target_include_directories(
    saltatlas INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:include>
)
target_link_libraries(
    saltatlas INTERFACE ${SALTATLAS_CEREAL_TARGET} ygm::ygm hnswlib::hnswlib
)
option(SALTATLAS_DETERMINISM "Avoid shuffling for deterministic output" OFF)
if (SALTATLAS_DETERMINISM)
    message(STATUS ${PROJECT_NAME} " setting determinism")
    target_compile_definitions(saltatlas INTERFACE SALTATLAS_DETERMINISM)
endif ()

#
# saltatlas_h5_io (needs findable HDF5 >= v1.10.7)
#
option(SALTATLAS_USE_HDF5 "Include HDF5 and h5 read/write utilities" ON)
if (SALTATLAS_USE_HDF5)
    find_package(saltatlas_h5_io CONFIG)
    if (NOT saltatlas_h5_io_FOUND)
        FetchContent_Declare(
            saltatlas_h5_io
            GIT_REPOSITORY https://github.com/LLNL/saltatlas_h5_io
            GIT_TAG main
        )
        FetchContent_GetProperties(saltatlas_h5_io)
        if (saltatlas_h5_io_POPULATED)
            message(
                STATUS ${PROJECT_NAME}
                       "found already populated saltatlas_h5_io dependency: "
                       ${h5_cpp_io_SOURCE_DIR}
            )
        else ()
            # Do not compile any saltatlas_h5_io tests
            set(JUST_INSTALL_SALTATLAS_H5_IO ON)
            # Install h5_cpp_io at ${CMAKE_INSTALL_PREFIX}
            set(SALTATLAS_H5_IO_INSTALL ${SALTATLAS_INSTALL})
            # Populate saltatlas_h5_io
            FetchContent_Populate(saltatlas_h5_io)
            # Include saltatlas_h5_io root cmake boilerplate
            add_subdirectory(
                ${saltatlas_h5_io_SOURCE_DIR} ${saltatlas_h5_io_BINARY_DIR}
            )
            message(STATUS ${PROJECT_NAME}
                           " cloned saltatlas_h5_io dependency: "
                           ${saltatlas_h5_io_SOURCE_DIR}
            )
        endif ()
    else ()
        message(STATUS ${PROJECT_NAME} " found saltatlas_h5_io dependency: "
                       ${saltatlas_h5_io_DIR}
        )
        if (${saltatlas_h5_io_SOURCE_DIR} MATCHES "*_deps*")
            add_subdirectory(
                ${saltatlas_h5_io_SOURCE_DIR} ${saltatlas_h5_io_BINARY_DIR}
            )
        endif ()
    endif ()
    target_link_libraries(saltatlas INTERFACE saltatlas_h5_io::saltatlas_h5_io)
else ()
    message(STATUS ${PROJECT_NAME} " building without saltatlas_h5_io and HDF5")
endif ()

#
# Metall
#
if (SALTATLAS_USE_METALL)
    find_package(Metall QUIET CONFIG)
    if (NOT Metall_FOUND)
        FetchContent_Declare(
            Metall
            GIT_REPOSITORY https://github.com/LLNL/metall.git
            GIT_TAG master
        )
        FetchContent_GetProperties(Metall)
        if (Metall_POPULATED)
            message(STATUS ${PROJECT_NAME}
                           " found already populated Metall dependency: "
                           ${metall_SOURCE_DIR}
            )
        else ()
            # Do not compile anything in Metall
            set(JUST_INSTALL_METALL_HEADER ON)
            FetchContent_MakeAvailable(Metall)
            message(STATUS ${PROJECT_NAME} " cloned Metall dependency "
                           ${metall_SOURCE_DIR}
            )
        endif ()
    else ()
        message(STATUS ${PROJECT_NAME} " found Metall dependency: "
                       ${Metall_INCLUDE_DIRS}
        )
    endif ()

    find_package(Threads REQUIRED)
    target_include_directories(
        saltatlas INTERFACE $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
                            $<INSTALL_INTERFACE:include/boost>
    )
    target_link_libraries(
        saltatlas INTERFACE stdc++fs Threads::Threads Metall::Metall
    )
endif ()

option(TEST_WITH_SLURM "Run tests with Slurm" OFF)

if (SALTATLAS_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(SALTATLAS_EXPORT_TARGETS saltatlas)

    install(
        TARGETS ${SALTATLAS_EXPORT_TARGETS}
        EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(
        EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
    )

    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/saltatlas
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # create version file
    write_basic_package_version_file(
        "${PROJECT_NAME}ConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY ExactVersion
    )

    # create config file
    configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
    )

    install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
    )

endif ()

option(JUST_INSTALL_SALTATLAS "Skip executables" OFF)
if (JUST_INSTALL_SALTATLAS)
    return()
endif ()

if (SALTATLAS_MAIN_PROJECT)
    add_subdirectory(examples)
    add_subdirectory(test)
endif ()
