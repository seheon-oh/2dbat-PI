#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
try:
    from setuptools import setup, find_package
    setup
except ImportError:
    from distutils.core import setup
    setup

dir_path = os.path.dirname(os.path.realpath(__file__))

init_string = open(os.path.join(dir_path, 'src','__init__.py')).read()

try:
    import pypandoc
    with open('README.md', 'r') as f:
        txt = f.read()
    txt = re.sub('<[^<]+>', '', txt)
    long_description = pypandoc.convert(txt, 'rst', 'md')
except ImportError:
    long_description = open('README.md').read()

setup(
    name                = '2dbat',
    version             = '1.0.0',
    description         = 'rotation curve analysis',
    author              = 'Se-Heon Oh',
    author_email        = 'seheon.oh@sejong.ac.kr',
    url                 = 'https://github.com/seheon-oh/2dbat-PI',
    packages            = ["src"],
    keywords            = ['HI rotation curve analysis', 'nested sampling'],
    python_requires     = '>=3',
    install_requires     = [
	"aiosignal>=1.3.1",
	"appnope>=0.1.3",
	"astropy>=5.2.1",
	"asttokens>=2.4.1",
	"attrs>=22.2.0",
	"casa-formats-io>=0.2.1",
	"certifi>=2022.12.7",
	"charset-normalizer>=3.0.1",
	"click>=8.1.3",
	"cloudpickle>=2.2.1",
	"comm>=0.2.1",
	"contourpy>=1.0.7",
	"cycler>=0.11.0",
	"Cython>=3.0.5",
	"dask>=2023.1.0",
	"debugpy>=1.8.0",
	"decorator>=5.1.1",
	"distlib>=0.3.6",
	"dynesty>=2.0.3",
	"entrypoints>=0.4",
	"exceptiongroup>=1.2.0",
	"executing>=2.0.1",
	"filelock>=3.9.0",
	"fitsio>=1.1.8",
	"fonttools>=4.38.0",
	"frozenlist>=1.3.3",
	"fsspec>=2023.1.0",
	"grpcio>=1.51.1",
	"idna>=3.4",
	"imageio>=2.31.6",
	"ipykernel>=6.28.0",
	"ipyparallel>=8.6.1",
	"ipython>=8.19.0",
	"jax>=0.4.20",
	"jaxlib>=0.4.20",
	"jedi>=0.19.1",
	"joblib>=1.2.0",
	"jsonschema>=4.17.3",
	"julia>=0.6.1",
	"jupyter_client>=8.6.0",
	"jupyter_core>=5.7.0",
	"kiwisolver>=1.4.4",
	"lazy_loader>=0.3",
	"llvmlite>=0.39.1",
	"locket>=1.0.0",
	"matplotlib>=3.6.3",
	"matplotlib-inline>=0.1.6",
	"ml-dtypes>=0.3.1",
	"msgpack>=1.0.4",
	"nest-asyncio>=1.5.8",
	"networkx>=3.2.1",
	"numba>=0.56.4",
	"numpy>=1.22.4",
	"opt-einsum>=3.3.0",
	"packaging>=23.0",
	"pandas>=2.1.4",
	"parso>=0.8.3",
	"partd>=1.3.0",
	"pexpect>=4.9.0",
	"Pillow>=9.4.0",
	"platformdirs>=2.6.2",
	"prompt-toolkit>=3.0.43",
	"protobuf>=4.21.12",
	"psutil>=5.9.4",
	"ptyprocess>=0.7.0",
	"pure-eval>=0.2.2",
	"pyerfa>=2.0.0.1",
	"Pygments>=2.17.2",
	"pyjulia>=0.0.6",
	"pyparsing>=3.0.9",
	"pyrsistent>=0.19.3",
	"python-dateutil>=2.8.2",
	"pytz>=2023.3.post1",
	"PyYAML>=6.0",
	"pyzmq>=25.1.2",
	"radio-beam>=0.3.4",
	"ray>=2.2.0",
	"requests>=2.28.2",
	"scikit-image>=0.22.0",
	"scipy>=1.11.3",
	"six>=1.16.0",
	"spectral-cube>=0.6.0",
	"stack-data>=0.6.3",
	"tifffile>=2023.9.26",
	"tk>=0.1.0",
	"toolz>=0.12.0",
	"tornado>=6.4",
	"tqdm>=4.66.1",
	"traitlets>=5.14.1",
	"tzdata>=2023.4",
	"urllib3>=1.26.14",
	"virtualenv>=20.17.1",
	"wcwidth>=0.2.13"
        ],
    package_data        = {"": ["README.md"]},
    include_package_data=True,
    zip_safe            = False,
    classifiers         = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
