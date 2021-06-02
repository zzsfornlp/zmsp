import os, sys
import glob
import platform

from distutils.core import setup, Extension
from distutils import sysconfig

if platform.system() == "Windows":
    cpp_args = ['-DNOMINMAX']
else:
    cpp_args = ['-std=c++11', '-O3', '-fPIC']

sfc_module = Extension(
    'parser2',
    sources=glob.glob("./ad3/*.cpp")+["algo.cpp", "parser2.cpp"],
    include_dirs=['pybind11/include', './ad3/', './Eigen/', '.'],
    language='c++',
    extra_compile_args=cpp_args,
    )

setup(
    name='parser2',
    ext_modules=[sfc_module],
)

# python setup.py build_ext
