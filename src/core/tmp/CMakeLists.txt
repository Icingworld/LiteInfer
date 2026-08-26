set(TMP_SOURCES
    tmp.cpp
)

add_library(liteinfer_tmp ${TMP_SOURCES})

target_include_directories(liteinfer_tmp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../)
