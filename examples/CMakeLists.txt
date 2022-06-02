# Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
# saltatlas Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

#
# This function adds an example.
#
function (add_saltatlas_example example_name)
    set(example_source "${example_name}.cpp")
    set(example_exe "${example_name}")
    add_executable(${example_exe} ${example_source})
    target_link_libraries(${example_exe} PUBLIC saltatlas)
endfunction ()

add_saltatlas_example(binary_nn_benchmark)
#add_saltatlas_example(knn_dpockets_vec_test)
#add_saltatlas_example(knn_dpockets_array_test)
add_saltatlas_example(space_wrapper_example)
add_saltatlas_example(saltatlas_example)

if (SALTATLAS_USE_HDF5)
    add_saltatlas_example(benchmark_hdf5)
	#add_saltatlas_example(knn_dpockets_hdf5)
endif ()
