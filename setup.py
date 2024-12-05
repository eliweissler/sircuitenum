import os
import re
from setuptools import setup

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with open(os.path.join(HERE, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='Superconducting Circuit Enumeration',
    version='0.1',  # find_version('asdf', '__init__.py'),
    description='A library for enumerating superconducting circuits',
    url='https://github.com/eliweissler/sircuitenum',
    author='Eli Weissler, Mohit Bhat',
    author_email='eli.weissler@colorado.edu',
    license='GNU GPL-3.0',
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    python_requires='>3.9.0',
    install_requires=[
                    # enumeration dependencies
                    'numpy==1.26.*',
                    'networkx==3.4.*',
                    'scipy==1.13.*',
                    'sympy==1.13.*',
                    'pandas==2.2.*',
                    'SQcircuit==1.0.*',
                    'scqubits==4.2.*',
                    'schemdraw==0.19.*',
                    'func_timeout==4.3.*',
                    'antlr4-python3-runtime==4.11',
                    'matplotlib',
                    'tqdm',
    ],
    extras_require = {
        'full': [ # test dependencies
                    'flake8==7.1.*',
                    'pytest==8.3.*',
                    'pytest-cov==6.*',
                    'coverage==7.6.*',
                    "requests",

                    # doc dependencies
                    'sphinx',
                    'sphinx-autodoc-typehints',
                    'sphinx_rtd_theme',
                    'nbsphinx',
                    'myst-parser',

                    # for testing examples
                    'nbval'
                ]
    },
    packages=['sircuitenum'],
)

py_modules = []
