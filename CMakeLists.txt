cmake_minimum_required(VERSION 3.26)

project(gpjson-cpp
  VERSION 0.1.0
  DESCRIPTION "C++/CUDA implementation of GpJSON"
  LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

option(GPJSON_CPP_DEBUG "Enable GPJSON debug logging and assertions" OFF)

add_library(gpjson-cpp_lib
  src/gpjson/cuda/cuda.cpp
  src/gpjson/engine.cpp
  src/gpjson/file/file.cpp
  src/gpjson/file/file_reader.cpp
  src/gpjson/index/index.cpp
  src/gpjson/index/index_builder.cpp
  src/gpjson/query/query.cpp
  src/gpjson/query/query_compiler.cpp
  src/gpjson/query/query_executor.cpp
)

target_include_directories(gpjson-cpp_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(gpjson-cpp_lib PUBLIC cxx_std_20)

if(GPJSON_CPP_DEBUG)
  target_compile_definitions(gpjson-cpp_lib PUBLIC GPJSON_CPP_DEBUG)
endif()

target_compile_options(gpjson-cpp_lib
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

set_target_properties(gpjson-cpp_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(gpjson-cpp_lib PUBLIC CUDA::cudart)

add_executable(gpjson-cpp apps/main.cpp)
target_link_libraries(gpjson-cpp PRIVATE gpjson-cpp_lib)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
