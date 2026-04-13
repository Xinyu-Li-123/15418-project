# add_executable(gpjson-cpp_tests
#   test_gpjson_dummy.cpp
# )
#
# target_link_libraries(gpjson-cpp_tests
#   PRIVATE
#     gpjson-cpp_lib
# )
#
# add_test(NAME gpjson-cpp_tests COMMAND gpjson-cpp_tests)
#
# find_package(GTest REQUIRED)
#
# add_executable(gpjson-cpp_cuda_tests
#   test_cuda.cpp
# )
#
# target_link_libraries(gpjson-cpp_cuda_tests
#   PRIVATE
#     gpjson-cpp_lib
#     GTest::gtest_main
# )
#
# add_test(NAME gpjson-cpp_cuda_tests COMMAND gpjson-cpp_cuda_tests)
#
#

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
#
# set_target_properties(peakfinder_tests PROPERTIES
#   CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
# )

include(GoogleTest)
gtest_discover_tests(gpjson-cuda_tests)
