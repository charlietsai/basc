Bayesian Active Site Calculator (BASC)
======================================

This repository contains the source code behind the Bayesian Active Site
Calculator (BASC).  BASC is a framework for rapidly and accurately searching
for the global minimum on potential energy surfaces.

## Dependencies

In order to run BASC, you need a working installation of:

1. [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/),
    available in the yum/apt-get package "python-ase"
2. [GPAW](https://wiki.fysik.dtu.dk/gpaw/),
    available in the yum/apt-get package "gpaw"
3. [GPy](https://github.com/SheffieldML/GPy),
    available in the pip package "gpy"
4. A Python runtime that supportes these packages (e.g. Python 2.7)

I recommend running BASC on Linux, although it should run on any platform that
supports these dependencies.

