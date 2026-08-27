set(LITEINFER_CORE_EMBEDDING_SOURCES
    embedding.cpp
)

add_library(liteinfer_core_embedding STATIC ${LITEINFER_CORE_EMBEDDING_SOURCES})

target_include_directories(liteinfer_core_embedding
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

target_link_libraries(liteinfer_core_embedding
    PUBLIC
        liteinfer_core_tensor
)
