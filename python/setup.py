from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='xlstm-cuda',
    version='0.1',
    packages=['xlstm', 'xlstm.layers', 'xlstm.blocks', 'xlstm.models'],
    ext_modules=[
        cpp_extension.CppExtension(
            'xlstm_cpp',
            ['../cpp/utils/utils.cpp',
             '../cpp/layers/slstm_layer.cpp',
             '../cpp/layers/mlstm_layer.cpp',
             '../cpp/blocks/xlstm_block.cpp',
             '../cpp/models/xlstm_model.cpp'],
            include_dirs=['..'],
            extra_compile_args=['-std=c++14'],
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    author='Stephen Vorwerk',
    author_email='smvorwerk@gmail.com'
)