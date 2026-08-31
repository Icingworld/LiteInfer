set(LITEINFER_CORE_SAFETENSOR_SOURCES
    safetensor_file.cpp
)

add_library(liteinfer_core_safetensor STATIC ${LITEINFER_CORE_SAFETENSOR_SOURCES})

target_include_directories(liteinfer_core_safetensor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

target_link_libraries(liteinfer_core_safetensor
    PUBLIC
        liteinfer_core_filesystem
    PRIVATE
        liteinfer_core_io
        liteinfer_core_json
)
