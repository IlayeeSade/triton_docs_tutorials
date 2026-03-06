from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='w4a16_cuda_ext',
    ext_modules=[
        CUDAExtension('w4a16_cuda_ext', [
            'w4a16_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })