add_library(liteinfer_core_tokenizer INTERFACE)

target_include_directories(liteinfer_core_tokenizer
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

target_sources(liteinfer_core_tokenizer
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_error.hpp
)

add_subdirectory(qwen3)
