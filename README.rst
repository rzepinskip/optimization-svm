optsvm
======

SVM implementation as quadaratic program by dual-task definition for Optimization Methods in Data Analysis classes.

Usage
-----

1. Run SVM training for all datasets and all solvers::

    python opt.py

Modify the beginning of `opt.py` file to specify subsets of datasets/solvers to run.

Installation
------------

1. Create a virtual environment::

    python -m venv .env

2. Activate the virtual environment::

    source .env/bin/activate

3. Install dependencies::

    pip install -r requirements.txt

Requirements
^^^^^^^^^^^^

Python 3.7

Authors
-------

`optsvm` was written by `Zuzanna Magierska, Paweł Rzepiński`.
