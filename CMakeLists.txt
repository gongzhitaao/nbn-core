cmake_minimum_required (VERSION 2.8)

project (nbn)

option(build_test "Build all of NBN's unit tests." ON)

include_directories(/usr/include/eigen3)
set(CMAKE_CXX_FLAGS "${cmake_cxx_flags} -O3 -std=c++11 -fopenmp")

add_subdirectory(src)

if (build_test)
  enable_testing()
  find_package(GTest REQUIRED)
  add_subdirectory(test)
endif()
