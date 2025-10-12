if (SALTATLAS_USE_METALL)
    add_executable(dnnd_transfer_pm_datastore dnnd_transfer_pm_datastore.cpp)
    target_link_libraries(dnnd_transfer_pm_datastore PUBLIC saltatlas)

    add_executable(dnnd_add_knng_edges dnnd_add_knng_edges.cpp)
    target_link_libraries(dnnd_add_knng_edges PUBLIC saltatlas)
endif ()
