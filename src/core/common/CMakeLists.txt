set(LITEINFER_CORE_COMMON_SOURCES
    data_type/float16.cpp
    data_type/bfloat16.cpp
)

add_library(liteinfer_core_common STATIC ${LITEINFER_CORE_COMMON_SOURCES})

target_include_directories(liteinfer_core_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../
)
