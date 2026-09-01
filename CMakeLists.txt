cmake_minimum_required(VERSION 4.2.3)
project(liteinfer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 禁用模块扫描
# 本项目暂不使用 module 特性
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

enable_testing()

add_subdirectory(src/core/tensor)
add_subdirectory(src/core/kvcache)
add_subdirectory(src/core/embedding)
add_subdirectory(src/core/kernels)
add_subdirectory(src/core/layer)
add_subdirectory(src/core/filesystem)
add_subdirectory(src/core/io)
add_subdirectory(src/core/json)
add_subdirectory(src/core/safetensors)
add_subdirectory(src/core/tokenizer)
add_subdirectory(src/core/model)
add_subdirectory(src/core/runtime)
add_subdirectory(examples)
add_subdirectory(tests)
