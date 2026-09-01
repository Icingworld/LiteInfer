set(LITEINFER_CORE_KVCACHE_SOURCES
    kvcache.cpp
)

add_library(liteinfer_core_kvcache STATIC ${LITEINFER_CORE_KVCACHE_SOURCES})

target_include_directories(liteinfer_core_kvcache
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

target_link_libraries(liteinfer_core_kvcache
    PUBLIC
        liteinfer_core_tensor
)
