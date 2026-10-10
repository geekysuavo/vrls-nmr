"""Main setup script, builds CUDA/C++ extension."""

# Third party
import pybind11
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext = CUDAExtension(
    name="vrlsnmr._ops",
    language="c++",
    include_dirs=[
        "include",
        pybind11.get_include(),
    ],
    sources=[
        "src/lib.cc",
        "src/impl-cpu.cc",
        "src/impl-cuda.cu",
    ],
    extra_compile_args={"cxx": ["-fopenmp"]},
)

build_ext = BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})
