[![Documentation Status](https://readthedocs.org/projects/sircuitenum/badge/?version=latest)](https://sircuitenum.readthedocs.io/en/latest/?badge=latest)
Superconducting cIRCUIT-ENUMeration
======================================

A library for enumerating superconducting circuits and exporting them for analysis using either [SQcircuit](https://github.com/stanfordLINQS/SQcircuit) or [scQubits](https://github.com/scqubits/scqubits). Used to produce the results in [Enumeration of all superconducting circuits up to 5 nodes](https://doi.org/10.48550/arXiv.2410.18497). For access to the databases of enumerated circuits used in the publication, please email the authors.

Installation
------------

Currently `sircuitenum` is installed from source. The tested paper environment
is defined in `environment-paper.yml` and includes Python, SageMath, Singular,
the numerical dependencies, JupyterLab, and the pinned SQcircuit revision.

### Paper environment

Clone the repository, check out the desired release or tagged revision, and
create the environment from the repository root:

```bash
git clone https://github.com/eliweissler/sircuitenum.git
cd sircuitenum/
git checkout <release-tag>
conda env create --solver libmamba -f environment-paper.yml
conda activate sircuitenum-paper
```

The environment recipe installs the checked-out source tree in editable mode,
so run the creation command from the `sircuitenum` repository root. No separate
`pip install` step is required. To recreate the environment, remove the old
environment first and run the same creation command again.

Start the notebook interface with:

```bash
conda activate sircuitenum-paper
jupyter lab
```

Examples
------------
Examples of how to use the library to enumerate and optimize circuits are included in the examples folder.


Testing
-------

The paper environment includes pytest. From the repository root, run the full
test suite with:

```bash
conda activate sircuitenum-paper
pytest -q
```

The complete suite includes computationally expensive enumeration and algebra
tests and can take approximately twenty minutes.

Disclaimer
----------

This package is currently in alpha (v0.x), and therefore you should not expect that APIs
will necessarily be stable between releases. Code that depends on this package in its current
state is very likely to break when the package version changes.

The paper environment pins the exact SQcircuit revision needed for the circuits
affected by the negligible charge-mode tolerance fix.


Runtime
----------

It is possible to consider four node circuits with a personal computer, but more resources are needed to enumerate and categorize five node circuits. We required about 50,000 core hours.
