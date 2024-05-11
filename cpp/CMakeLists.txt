cmake_minimum_required(VERSION 3.8)
project(xlstm_cpp)

# Set C++ compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -std=c++14")

# Include directories
include_directories(utils)
include_directories(layers)
include_directories(blocks)
include_directories(models)

# Source files
set(XLSTM_SOURCES
    utils/utils.cpp
    layers/slstm_layer.cpp
    layers/mlstm_layer.cpp
    blocks/xlstm_block.cpp
    models/xlstm_model.cpp
)

# Library target
add_library(xlstm_cpp STATIC ${XLSTM_SOURCES})

# Specify include directories for the library target
target_include_directories(xlstm_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link against the CUDA library
target_link_libraries(xlstm_cpp xlstm_cuda)