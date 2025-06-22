from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext = CUDAExtension(
    name="vrlsnmr._ops",
    language="c++",
    include_dirs=["include"],
    sources=[
        "src/lib.cc",
        "src/impl-cpu.cc",
        "src/impl-cuda.cu",
    ],
    extra_compile_args={"cxx": ["-fopenmp"]},
)

build_ext = BuildExtension.with_options(
    no_python_abi_suffix=True,
    use_ninja=False,
)

setup(
    name="vrlsnmr",
    version="0.0.1",
    author="Bradley Worley",
    author_email="geekysuavo@gmail.com",
    packages=find_packages(),
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
