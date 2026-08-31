set(LITEINFER_CORE_FILESYSTEM_SOURCES
    file_handle.cpp
    filesystem.cpp
)

if(UNIX)
    list(APPEND LITEINFER_CORE_FILESYSTEM_SOURCES
        backend/posix/posix_file_handle_backend.cpp
        backend/posix/posix_filesystem_backend.cpp
    )
endif()

add_library(liteinfer_core_filesystem STATIC ${LITEINFER_CORE_FILESYSTEM_SOURCES})

target_include_directories(liteinfer_core_filesystem
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../
)
