add_executable(gpjson-cpp_tests
  test_gpjson_dummy.cpp
)

target_link_libraries(gpjson-cpp_tests
  PRIVATE
    gpjson-cpp_lib
)

add_test(NAME gpjson-cpp_tests COMMAND gpjson-cpp_tests)
