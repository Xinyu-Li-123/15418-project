include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE GPJSON_TESTS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_executable(gpjson-cuda_tests
  ${GPJSON_TESTS}
)

target_include_directories(gpjson-cuda_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(gpjson-cuda_tests
  PRIVATE
    gpjson-cpp_lib
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(gpjson-cuda_tests)
