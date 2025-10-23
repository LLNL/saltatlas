# Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
# saltatlas Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

#
# This function adds an example.
#
function(add_saltatlas_example example_name)
    set(example_source "${example_name}.cpp")
    set(example_exe "${example_name}")
    add_executable(${example_exe} ${example_source})
    target_link_libraries(${example_exe} PUBLIC saltatlas Boost::unordered Boost::container)
endfunction()

add_saltatlas_example(dhnsw_bench)

#
# This function adds a DNND example with the given feature element type.
#
function(add_saltatlas_dnnd_example_feature_type example_name type_name)
    set(example_source "${example_name}.cpp")
    set(example_exe "${example_name}_${type_name}_features")
    add_executable(${example_exe} ${example_source})
    target_link_libraries(${example_exe} PUBLIC saltatlas Boost::unordered Boost::container)
    target_compile_definitions(
        ${example_exe}
        PRIVATE "SALTATLAS_DNND_EXAMPLE_FEATURE_ELEMENT_TYPE=${type_name}"
    )
endfunction()

#
# This function adds a NEO-DNND example
#
function(add_saltatlas_neo_dnnd_example example_name)
    set(example_exe "${example_name}")
    add_executable(${example_exe} "${example_name}.cpp")
    setup_mpi_target(${example_exe})
    target_link_libraries(${example_exe} PUBLIC saltatlas)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(${example_exe} PUBLIC OpenMP::OpenMP_CXX)
    else ()
        message(WARNING "OpenMP not found, ${example_exe} will not be built with OpenMP support")
    endif ()
endfunction()

#
# This function adds a NEO-DNND example with the given feature element type.
#
function(add_saltatlas_neo_dnnd_example_feature_type example_name type_name)
    set(example_exe "${example_name}_${type_name}_features")
    add_executable(${example_exe} "${example_name}.cpp")
    setup_mpi_target(${example_exe})
    target_link_libraries(${example_exe} PUBLIC saltatlas)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(${example_exe} PUBLIC OpenMP::OpenMP_CXX)
    else ()
        message(WARNING "OpenMP not found, ${example_exe} will not be built with OpenMP support")
    endif ()
    target_compile_definitions(${example_exe}
        PRIVATE "SALTATLAS_NEO_DNND_EXAMPLE_FEATURE_ELEMENT_TYPE=${type_name}")
endfunction()

add_saltatlas_example(dnnd_simple)
add_saltatlas_example(dnnd_simple_custom_distance)
add_saltatlas_example(dnnd_simple_custom_point)
add_saltatlas_example(dnnd_simple_levenshtein)
add_saltatlas_example(dnnd_simple_charhist)
add_saltatlas_example(dnnd_simple_levenshtein_utf8)

add_saltatlas_dnnd_example_feature_type(dnnd_bench float)
add_saltatlas_dnnd_example_feature_type(dnnd_bench uint8_t)
add_saltatlas_dnnd_example_feature_type(dnnd_bench double)

if (SALTATLAS_USE_METALL)
    add_saltatlas_example(dnnd_advanced)
    add_saltatlas_example(dnnd_advanced_index_build)
    add_saltatlas_example(dnnd_advanced_get_neighbors)
    add_saltatlas_example(dnnd_advanced_query)
    add_saltatlas_example(dnnd_advanced_timeout)
    
    add_saltatlas_neo_dnnd_example(neo_dnnd_example)
    add_saltatlas_neo_dnnd_example(neo_dnnd_initial_knng_example)

    add_saltatlas_neo_dnnd_example_feature_type(neo_dnnd_bench float)
    add_saltatlas_neo_dnnd_example_feature_type(neo_dnnd_bench uint8_t)
    add_saltatlas_neo_dnnd_example_feature_type(neo_dnnd_bench double)
endif ()

#
# This function adds a shared-memory executable with the given feature element type.
#
function(add_fe_typed_shm_exe target_name type_name)
    set(typed_target_name "${target_name}_${type_name}_features")

    add_executable(${typed_target_name} "${target_name}.cpp")

    target_include_directories(
        ${typed_target_name}
        PRIVATE ${CMAKE_SOURCE_DIR}/include
    )

    setup_omp_target(${typed_target_name})
    target_link_libraries(${typed_target_name} PRIVATE ${BOOST_LIBS})

    target_compile_definitions(
        ${typed_target_name}
        PRIVATE "SALTATLAS_FEATURE_ELEMENT_TYPE=${type_name}"
    )
    set(target_name
        ${typed_target_name}
        PARENT_SCOPE
    )

    setup_metall_target(${typed_target_name})
endfunction()

add_fe_typed_shm_exe(run_query_knng float)
add_fe_typed_shm_exe(run_query_knng double)
add_fe_typed_shm_exe(run_query_knng uint8_t)


add_subdirectory(datasets)
#add_subdirectory(knng_shm_query)