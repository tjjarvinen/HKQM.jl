# HKQM (Helmholtz Kernel Quantum Mechanics)

| **Documentation**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-dev-img]][docs-dev-url] | [![][CI-img]][CI-url] [![][codecov-img]][codecov-url] |




This package is under development and not all of the functionality is ready.
You are free to try it though.

The name is a working name and will change for the final program.

## What is it?

This is an electronic structure code on isolated molecules.
It uses new method to solve scf-equations (Helmholtz equation Greens function) and Kohn-Sham equations.

The program is designed so that it will have massive parallelization on GPUs,
while still being more simple than standard Gaussian orbital based methods.

The code is based on earlier Fortran code [DAGE](https://github.com/dagesundholm/DAGE) described in [chem.phys. 146, 084102 (2017)](http://dx.doi.org/10.1063/1.4976557).

This Julia based next generation implementation includes upgrades for accuracy, advanced functionality like automatic differentiation and general operator manipulations, while also being significantly faster and able to run on GPUs from multiple vendors.

## Development Plan

The aim is to run this code on [LUMI](https://www.lumi-supercomputer.eu/) supercomputer.

The plan is that the program is functional with DFT using pseudopotentials.

Later on we will add support for all electron calculations with [Bubbles](http://dx.doi.org/10.1021/acs.jctc.8b00456) framework.

## How Does it Work

The main idea here is to transform 3D Coulomb Integral to four one dimensional
integrals and thus massively reduce the computational time. This allows solving
Poisson equation. Helmholtz equation can then be solved using Greens function
the same way as Poisson equation, by only adding an extra constant term.

From the 4 one dimensional integrals 3 can be calculated as a tensor contraction
that parallelizes well on GPUs. The final integral can be parallelized over different GPUs. This should result in a program that can parallelize over tens of GPUs, for large systems.

THe main method also means that differential equation is solved as integral equation.
Thus the basis set is chosen to give the best accuracy for general numerical
integrals, which means Gauss-Legendre polynomials. The system is also divided to
elements in order to reduce the maximum order of Gauss-Legendre polynomials.

## Highlighted Features (when complete)

- Solve Poisson equation (3D) - ready
- Solve Helmholtz equation (3D) - ready
- Solve Schrödinger equation with Helmholtz kernel Greens function - (1 particle) ready
- General Hartree-Fock calculation (including electronic structure) - ready
- Parallelization on CPUs across different nodes - ready
- GPU calculations - Nvidia, AMD, Intel and Apple GPUs work
- Solve electronic structure with DFT - (needs KS Hamiltonian implementation and XC functionals that work with GPU, but CPU should work with libxc)
- Calculate magnetic field effects on electronic structure - (HF) ready
- Full automatic differentiation support - (forward mode working with TensorOperations backend, reverse needs special pullbacks for tensor contractions)

## Note

Not yet ready for general use. There is an issue with
integral accuracy. Meaning that Helmholtz equation
accuracy is about 1E-4, which is not enough for solving
Schrödinger equation. There is a fix on this incoming.



[CI-img]: https://github.com/tjjarvinen/HKQM.jl/workflows/CI/badge.svg
[CI-url]: https://github.com/tjjarvinen/HKQM.jl/actions?query=workflow%3ACI

[codecov-img]: https://codecov.io/gh/tjjarvinen/HKQM.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/tjjarvinen/HKQM.jl

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://tjjarvinen.github.io/HKQM.jl/dev/
