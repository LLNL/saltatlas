configure_file(convert_ann_bench_dataset.py convert_ann_bench_dataset.py COPYONLY)
configure_file(convert_big_ann_bench_dataset.py convert_big_ann_bench_dataset.py COPYONLY)

#  TODO: make compilable
# add_executable(convert_wsv_to_wsv_id convert_wsv_to_wsv_id.cpp)
# target_link_libraries(convert_wsv_to_wsv_id PUBLIC saltatlas)
# setup_omp_target(convert_wsv_to_wsv_id)

add_fe_typed_ygm_exe(convert_wsv_id_to_bin_id float)
add_fe_typed_ygm_exe(convert_wsv_id_to_bin_id uint8_t)

add_fe_typed_ygm_exe(gen_shifted_dataset float)
add_fe_typed_ygm_exe(gen_shifted_dataset uint8_t)
