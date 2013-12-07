include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(test_nbn test_nbn.cpp)

target_link_libraries(test_nbn gtest gtest_main)
target_link_libraries(test_nbn pthread)
target_link_libraries(test_nbn nbn)

add_test(test-nbn test_nbn)
