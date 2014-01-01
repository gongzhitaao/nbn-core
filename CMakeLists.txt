cmake_minimum_required (VERSION 2.8)

project (nbn-core)

option(build_test "Build all of NBN's unit tests (GTest required.)." ON)

include_directories(/usr/include/eigen3)
<<<<<<< Updated upstream
set(CMAKE_CXX_FLAGS "${cmake_cxx_flags} -Wall -O3 -std=c++11 -fopenmp")

add_subdirectory(src)

if (build_test)
  enable_testing()
  find_package(GTest REQUIRED)
  add_subdirectory(test)
endif()
