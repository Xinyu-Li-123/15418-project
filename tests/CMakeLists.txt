include(FetchContent)

if(NOT TARGET GTest::gtest_main)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
  )
  FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

function(gpjson_configure_test_target target_name)
  target_include_directories(${target_name}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/include
      ${PROJECT_SOURCE_DIR}/src
  )
  set_target_properties(${target_name} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
  gtest_discover_tests(${target_name})
endfunction()

file(GLOB GPJSON_INDEX_TESTS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/index/*.cpp"
)

add_executable(gpjson-cuda_tests
  cuda/test_cuda.cpp
)
target_link_libraries(gpjson-cuda_tests
  PRIVATE
    gpjson-cpp_lib
    GTest::gtest_main
)
gpjson_configure_test_target(gpjson-cuda_tests)

add_executable(gpjson-file_reader_tests
  file/file_reader_test.cpp
  ${PROJECT_SOURCE_DIR}/src/gpjson/file/file.cpp
  ${PROJECT_SOURCE_DIR}/src/gpjson/file/file_reader.cpp
)
target_link_libraries(gpjson-file_reader_tests
  PRIVATE
    GTest::gtest_main
)
gpjson_configure_test_target(gpjson-file_reader_tests)

add_executable(gpjson-index_tests
  ${GPJSON_INDEX_TESTS}
)
target_link_libraries(gpjson-index_tests
  PRIVATE
    gpjson-cpp_lib
    GTest::gtest_main
)
gpjson_configure_test_target(gpjson-index_tests)

add_executable(gpjson-engine_tests
  engine_test.cpp
)
target_link_libraries(gpjson-engine_tests
  PRIVATE
    gpjson-cpp_lib
    GTest::gtest_main
)
gpjson_configure_test_target(gpjson-engine_tests)

add_executable(gpjson-benchmark_output_tests
  benchmark/benchmark_output_test.cpp
)
target_link_libraries(gpjson-benchmark_output_tests
  PRIVATE
    gpjson-cpp_lib
    GTest::gtest_main
)
target_compile_definitions(gpjson-benchmark_output_tests
  PRIVATE
    GPJSON_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    GPJSON_BENCHMARK_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/tests/benchmark/output"
)
gpjson_configure_test_target(gpjson-benchmark_output_tests)

add_executable(gpjson-benchmark_large_output_tests
  benchmark/benchmark_large_output_test.cpp
)
target_link_libraries(gpjson-benchmark_large_output_tests
  PRIVATE
    gpjson-cpp_lib
    GTest::gtest_main
)
target_compile_definitions(gpjson-benchmark_large_output_tests
  PRIVATE
    GPJSON_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    GPJSON_BENCHMARK_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/tests/benchmark/output"
)
gpjson_configure_test_target(gpjson-benchmark_large_output_tests)

add_executable(gpjson-benchmark_remove_output_tests
  benchmark/benchmark_remove_output_test.cpp
)
target_link_libraries(gpjson-benchmark_remove_output_tests
  PRIVATE
    gpjson-cpp_lib
    GTest::gtest_main
)
target_compile_definitions(gpjson-benchmark_remove_output_tests
  PRIVATE
    GPJSON_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    GPJSON_BENCHMARK_OUTPUT_DIR="${PROJECT_SOURCE_DIR}/tests/benchmark/output"
)
gpjson_configure_test_target(gpjson-benchmark_remove_output_tests)

add_executable(gpjson-query_compiler_tests
  query/query_compiler_test.cpp
  ${PROJECT_SOURCE_DIR}/src/gpjson/query/query.cpp
  ${PROJECT_SOURCE_DIR}/src/gpjson/query/query_compiler.cpp
  ${PROJECT_SOURCE_DIR}/src/gpjson/query/ir_input_buffer.cpp
)
target_link_libraries(gpjson-query_compiler_tests
  PRIVATE
    GTest::gtest_main
)
gpjson_configure_test_target(gpjson-query_compiler_tests)

add_executable(gpjson-query_executor_tests
  query/query_executor_test.cpp
)
target_link_libraries(gpjson-query_executor_tests
  PRIVATE
    gpjson-cpp_lib
    GTest::gtest_main
)
gpjson_configure_test_target(gpjson-query_executor_tests)
