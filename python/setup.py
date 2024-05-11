from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='xlstm-cuda',
    version='0.1.0',
    description='Extended Long Short-Term Memory (xLSTM) Library',
    author='Stephen Vorwerk',
    author_email='smvorwerk@gmail.com',
    url='https://github.com/smvorwerk/xlstm-cuda',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            'xlstm_cpp',
            [
                'cpp/utils/utils.cpp',
                'cpp/layers/slstm_layer.cpp',
                'cpp/layers/mlstm_layer.cpp',
                'cpp/blocks/xlstm_block.cpp',
                'cpp/models/xlstm_model.cpp',
                'cuda/kernels/slstm_kernels.cu',
                'cuda/kernels/mlstm_kernels.cu',
                'cuda/kernels/block_kernels.cu',
            ],
            include_dirs=['cpp', 'cpp/utils', 'cpp/layers', 'cpp/blocks', 'cpp/models', 'cuda/kernels', 'cuda/utils'],
            extra_compile_args={
                'cxx': ['-std=c++14'],
                'nvcc': ['-arch=sm_60', '-std=c++14'],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
    install_requires=[
        'torch>=1.8.0',
        'numpy>=1.19.0',
    ],
)