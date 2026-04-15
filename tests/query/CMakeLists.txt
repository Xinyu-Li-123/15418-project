if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(gpjson-query-compiler-tests LANGUAGES CXX)
  include(CTest)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(NOT TARGET GTest::gtest_main)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
  )
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(gpjson-query_compiler_tests
  query_compiler_test.cpp
  ../../src/gpjson/query/query.cpp
  ../../src/gpjson/query/query_compiler.cpp
  ../../src/gpjson/query/ir_input_buffer.cpp
)

target_include_directories(gpjson-query_compiler_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(gpjson-query_compiler_tests
  PRIVATE
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(gpjson-query_compiler_tests)
