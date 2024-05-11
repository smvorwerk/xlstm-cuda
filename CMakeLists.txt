cmake_minimum_required(VERSION 3.8)
project(xLSTM)

# Set C++ compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -std=c++14")

# Include directories
include_directories(cpp)
include_directories(cpp/utils)
include_directories(cpp/layers)
include_directories(cpp/blocks)
include_directories(cpp/models)

# Recurse into the CUDA and C++ subdirectories
add_subdirectory(cuda)
add_subdirectory(cpp)

# Executable targets for tests
add_executable(test_slstm tests/test_slstm.cpp)
target_link_libraries(test_slstm xlstm_cpp)

add_executable(test_mlstm tests/test_mlstm.cpp)
target_link_libraries(test_mlstm xlstm_cpp)

add_executable(test_xlstm_block tests/test_xlstm_block.cpp)
target_link_libraries(test_xlstm_block xlstm_cpp)

add_executable(test_xlstm_model tests/test_xlstm_model.cpp)
target_link_libraries(test_xlstm_model xlstm_cpp)

# Executable targets for examples
add_executable(example_slstm examples/example_slstm.cpp)
target_link_libraries(example_slstm xlstm_cpp)

add_executable(example_mlstm examples/example_mlstm.cpp)
target_link_libraries(example_mlstm xlstm_cpp)

add_executable(example_xlstm_block examples/example_xlstm_block.cpp)
target_link_libraries(example_xlstm_block xlstm_cpp)

add_executable(example_xlstm_model examples/example_xlstm_model.cpp)
target_link_libraries(example_xlstm_model xlstm_cpp)