#!/bin/bash

# Set the paths to the source directories
CUDA_DIR="cuda"
CPP_DIR="cpp"
PYTHON_DIR="python"

# Build and install the CUDA extension
cd $CUDA_DIR
mkdir -p build
cd build
cmake ..
make
cd ../..

# Build and install the C++ library
cd $CPP_DIR
mkdir -p build
cd build
cmake ..
make
cd ../..

# Install the Python package
cd $PYTHON_DIR
python setup.py install
cd ..

echo "xLSTM library built and installed successfully!"