# Add a target to generate API documentation with Doxygen
function(saltatlasDoxygen)
    find_package(Doxygen)
    if (NOT DOXYGEN_FOUND)
        message(WARNING "Doxygen not found, will skip Doxygen documentation generation.")
        return()
    endif()

    # Set the Doxygen configuration file
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_HTML_OUTPUT
            ${PROJECT_BINARY_DIR}/docs/html)
    set(DOXYGEN_EXAMPLE_PATH ${PROJECT_SOURCE_DIR}/examples)

    # Add a target 'saltatlas_doxygen'
    # Run 'make saltatlas_doxygen' to generate the docs
    doxygen_add_docs(saltatlas_doxygen
            ${PROJECT_SOURCE_DIR}/include
            COMMENT "Generate documentation by Doxygen"
            )
endfunction()