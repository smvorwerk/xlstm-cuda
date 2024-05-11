# Extended Long Short-Term Memory (xLSTM)

This repository contains the implementation of the Extended Long Short-Term Memory (xLSTM) architecture, as described in the paper [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517). xLSTM is an extension of the original LSTM architecture that aims to overcome some of its limitations while leveraging the latest techniques from modern large language models.

Read more about this library and the architecture it is attempting to implement in [OVERVIEW.md](OVERVIEW.md).

## Features

- Exponential gating with normalization and stabilization techniques
- Modified memory structures: sLSTM (scalar memory, scalar update, new memory mixing) and mLSTM (fully parallelizable, matrix memory, covariance update rule)
- xLSTM blocks that integrate sLSTM and mLSTM into residual block backbones
- xLSTM architectures constructed by residually stacking xLSTM blocks

## Requirements

- C++14 compiler
- CUDA toolkit
- CMake (version 3.8 or higher)
- Python (version 3.6 or higher)
- PyTorch (version 1.8 or higher)

## Installation

1. Clone the repository:

    ```shell
    git clone https://github.com/smvorwerk/xlstm-cuda.git
    cd xLSTM
    ```

2. Build the C++ and CUDA libraries:

    ```shell
    mkdir build
    cd build
    cmake ..
    make
    ```

3. Install the Python package:

    ```shell
    cd ../python
    python setup.py install
    ```

## Usage

### Time Series Model

xLSTM can be used as a powerful time series model due to its ability to capture long-term dependencies and handle complex temporal patterns. Here's an example of how to use xLSTM for time series forecasting:

```python
import torch
import torch.nn as nn
from xlstm import PyXLSTMModel

# Define the xLSTM model
input_size = 10
hidden_size = 64
proj_size = 32
use_mlstm_vec = [True, False, True]
num_layers = len(use_mlstm_vec)
model = PyXLSTMModel(input_size, hidden_size, proj_size, use_mlstm_vec, num_layers)

# Prepare the input data
seq_length = 100
batch_size = 32
input_data = torch.randn(batch_size, seq_length, input_size)

# Forward pass
output = model(input_data)

# Use the output for time series forecasting tasks
```

### Large Language Model

xLSTM can also be used as a language model, capable of generating coherent and contextually relevant text. Here's an example of how to use xLSTM for language modeling:

```python
import torch
import torch.nn as nn
from xlstm import PyXLSTMModel

# Define the xLSTM model
vocab_size = 10000
embedding_size = 128
hidden_size = 256
proj_size = 128
use_mlstm_vec = [True, False, True, False]
num_layers = len(use_mlstm_vec)
model = nn.Embedding(vocab_size, embedding_size)
model = PyXLSTMModel(embedding_size, hidden_size, proj_size, use_mlstm_vec, num_layers)

# Prepare the input data
seq_length = 50
batch_size = 16
input_data = torch.randint(0, vocab_size, (batch_size, seq_length))

# Forward pass
embeddings = model(input_data)
output = model(embeddings)

# Use the output for language modeling tasks
```

## Citation

If you use xLSTM in your research, please cite the original paper:

@article{beck2023xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and Pöppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, Günter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2023}
}

## License

This project is licensed under the [MIT License](./LICENSE).

## Directory Structure

```shell
xlstm/
├── cuda/
│   ├── kernels/
│   │   ├── slstm_kernels.cu
│   │   ├── mlstm_kernels.cu
│   │   └── block_kernels.cu
│   ├── utils/
│   │   └── cuda_utils.h
│   └── CMakeLists.txt
├── cpp/
│   ├── layers/
│   │   ├── slstm_layer.h
│   │   ├── slstm_layer.cpp
│   │   ├── mlstm_layer.h
│   │   └── mlstm_layer.cpp
│   ├── blocks/
│   │   ├── xlstm_block.h
│   │   └── xlstm_block.cpp
│   ├── models/
│   │   ├── xlstm_model.h
│   │   └── xlstm_model.cpp
│   ├── utils/
│   │   ├── utils.h
│   │   └── utils.cpp
│   ├── tests/
│   │   ├── test_slstm.cpp
│   │   ├── test_mlstm.cpp
│   │   ├── test_xlstm_block.cpp
│   │   └── test_xlstm_model.cpp
│   ├── examples/
│   │   ├── example_slstm.cpp
│   │   ├── example_mlstm.cpp
│   │   ├── example_xlstm_block.cpp
│   │   └── example_xlstm_model.cpp
│   └── CMakeLists.txt
├── python/
│   ├── xlstm/
│   │   ├── __init__.py
│   │   ├── layers/
│   │   │   ├── __init__.py
│   │   │   ├── slstm_layer.py
│   │   │   └── mlstm_layer.py
│   │   ├── blocks/
│   │   │   ├── __init__.py
│   │   │   └── xlstm_block.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── xlstm_model.py
│   │   └── utils/
│   │       └── __init__.py
│   └── setup.py
├── CMakeLists.txt
├── OVERVIEW.md
└── README.md
```
