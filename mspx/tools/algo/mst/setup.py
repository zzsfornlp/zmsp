# several files with ext .pyx
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy as np
import os

CUR_DIR = os.path.dirname(__file__)

extensions = [
    Extension("cmst",
              [os.path.join(CUR_DIR, "cmst.pyx")],
              include_dirs=[np.get_include()],
              # extra_compile_args=["-O3"],
              extra_compile_args=["-O2"],
              # fighting the anaconda issue if needed
              extra_link_args=['-L/usr/lib/x86_64-linux-gnu/'],
              ),
]

setup(
    name='MSPTasks',
    ext_modules=cythonize(extensions),
    options={
        # 'build_ext': {'inplace': True},
        'build': {'build_lib': CUR_DIR},
    }
)

# cython -a cmst.pyx
# python setup.py build_ext
