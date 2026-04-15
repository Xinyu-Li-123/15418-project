include(FetchContent)

if(NOT TARGET GTest::gtest_main)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
  )
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(gpjson-file_reader_tests
  file_reader_test.cpp
  ../../src/gpjson/file/file.cpp
  ../../src/gpjson/file/file_reader.cpp
)

target_include_directories(gpjson-file_reader_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(gpjson-file_reader_tests
  PRIVATE
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(gpjson-file_reader_tests)
