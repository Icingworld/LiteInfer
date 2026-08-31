set(LITEINFER_CORE_IO_SOURCES
    byte_reader.cpp
    buffer_byte_reader.cpp
    file_byte_reader.cpp
    binary_reader.cpp
)

add_library(liteinfer_core_io STATIC ${LITEINFER_CORE_IO_SOURCES})

target_include_directories(liteinfer_core_io
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

target_link_libraries(liteinfer_core_io
    PUBLIC
        liteinfer_core_filesystem
)
