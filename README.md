[![Documentation Status](https://readthedocs.org/projects/sircuitenum/badge/?version=latest)](https://sircuitenum.readthedocs.io/en/latest/?badge=latest)
Superconducting cIRCUIT-ENUMeration
======================================

A library for enumerating superconducting circuits.

At the moment this package requires a modified version of [SQcircuit](https://github.com/stanfordLINQS/SQcircuit/) to function. Hopefully, by January 2025, we will have the package working on the current versions of [SQcircuit](https://github.com/stanfordLINQS/SQcircuit/) and [scqubits](https://github.com/scqubits/scqubits).


Installation
------------

`sircuitenum` can be installed from source or via the Python package manager PyPI. The package has been tested on Python 3.11 on both Linux and Mac.

### Source

```bash
git clone https://github.com/combes-group/sircuitenum.git
cd sircuitenum/
pip install -e .
```

### PyPI

```bash
pip install sircuitenum
```

Examples
------------
Examples of how to use the library to enumerate and optimize circuits are included in the examples folder.

Library Philosophy
------------------

The core philosophy of `sircuitenum` is to be:

* Interoperable with different circuit quantization packages
* Separate data generation, data analysis, data visualisation

 
Testing
-------

The unit tests can be run locally using `pytest`. To install testing dependencies, install sircuitenum using

```bash
pip install sircuitenum[full]
```

Disclaimer
----------

This package is currently in alpha (v0.x), and therefore you should not expect that APIs
will necessarily be stable between releases. Code that depends on this package in its current
state is very likely to break when the package version changes, so we encourage you to pin
the version you use, and update it consciously when necessary.
