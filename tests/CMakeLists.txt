include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(gpjson-cuda_tests
  test_cuda.cpp
)

target_link_libraries(gpjson-cuda_tests
  PRIVATE
    gpjson-cpp_lib
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(gpjson-cuda_tests)
