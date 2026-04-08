cmake_minimum_required(VERSION 3.26)

project(peakfinder
  VERSION 0.1.0
  DESCRIPTION "Modern C++ + CUDA sample: local peak finding with Thrust scan"
  LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# For reproducible binaries, set architectures explicitly.
# RTX 2080 Ti is Turing, SM 75.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_library(peakfinder
  src/peak_utils.cpp
  src/peak_finder.cu
  src/peak_kernels.cu
)

target_include_directories(peakfinder
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(peakfinder PUBLIC cxx_std_20)

target_compile_options(peakfinder
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Native CUDA support is the modern path.
set_target_properties(peakfinder PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Thrust is header-only and comes with the CUDA Toolkit headers.
# We still locate the toolkit explicitly so imported CUDA targets are available.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(peakfinder PUBLIC CUDA::cudart)

add_executable(peakfinder_cli apps/main.cpp)
target_link_libraries(peakfinder_cli PRIVATE peakfinder)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
