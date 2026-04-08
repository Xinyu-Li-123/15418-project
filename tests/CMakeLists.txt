include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(peakfinder_tests
  test_peak_utils.cpp
  test_peak_cuda_pipeline.cpp
)

target_link_libraries(peakfinder_tests
  PRIVATE
    peakfinder
    GTest::gtest_main
)

set_target_properties(peakfinder_tests PROPERTIES
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

include(GoogleTest)
gtest_discover_tests(peakfinder_tests)
