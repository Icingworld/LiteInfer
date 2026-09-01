set(LITEINFER_CORE_SAFETENSORS_SOURCES
    safetensors_file.cpp
)

add_library(liteinfer_core_safetensors STATIC ${LITEINFER_CORE_SAFETENSORS_SOURCES})

target_include_directories(liteinfer_core_safetensors
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

target_link_libraries(liteinfer_core_safetensors
    PUBLIC
        liteinfer_core_filesystem
    PRIVATE
        liteinfer_core_io
        liteinfer_core_json
)
