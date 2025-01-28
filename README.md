[![Documentation Status](https://readthedocs.org/projects/sircuitenum/badge/?version=latest)](https://sircuitenum.readthedocs.io/en/latest/?badge=latest)
Superconducting cIRCUIT-ENUMeration
======================================

A library for enumerating superconducting circuits and exporting them for analysis using either [SQcircuit](https://github.com/stanfordLINQS/SQcircuit) or [scQubits](https://github.com/scqubits/scqubits). Used to produce the results in [Enumeration of all superconducting circuits up to 5 nodes](https://doi.org/10.48550/arXiv.2410.18497). For access to the databases of enumerated circuits used in the publication, please email the authors.

Installation
------------

Currently `sircuitenum` can only be installed from source, although installation via the Python package manager PyPI will be added soon. The package has been tested on Python 3.11 on both Linux and Mac.

### Source

```bash
git clone https://github.com/combes-group/sircuitenum.git
cd sircuitenum/
pip install -e .
```

Examples
------------
Examples of how to use the library to enumerate and optimize circuits are included in the examples folder.


Testing
-------

The unit tests can be run locally using `pytest`. To install testing dependencies, install sircuitenum using

```bash
pip install pip install -e .[full]
```

Disclaimer
----------

This package is currently in alpha (v0.x), and therefore you should not expect that APIs
will necessarily be stable between releases. Code that depends on this package in its current
state is very likely to break when the package version changes.

At the moment, a few circuits examined in the paper require a development branch of [SQcircuit](https://github.com/stanfordLINQS/SQcircuit/tree/dev-ew) to run properly.