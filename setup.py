from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name="hdrc",
    ext_modules=[
        CUDAExtension(
            name="hdrc._C",
            sources=[
            "utils\\utils.cu",
            "poisson_solvers\\solvers.cu",
            "hdrc.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
