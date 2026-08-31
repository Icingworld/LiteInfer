set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")

add_subdirectory(
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
    ${CMAKE_BINARY_DIR}/_deps/nlohmann_json
    EXCLUDE_FROM_ALL
)

add_library(liteinfer_core_json INTERFACE)

target_sources(liteinfer_core_json
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/json.hpp
)

target_include_directories(liteinfer_core_json
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

target_link_libraries(liteinfer_core_json
    INTERFACE
        nlohmann_json::nlohmann_json
)
