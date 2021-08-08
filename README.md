# HKQM (Helmholtz Kernel Quantum Mechanics)
[![][CI-img]][CI-url] [![][codecov-img]][codecov-url]


This package is under development and not ready for use yet!

The name is a working name and will change for the final program.

## What is it?

This is an electronic stucture code on isolated molecules.
It uses new method to solve scf-equations (Helmholtz equation Greens function) and Kohn-Sham equations.

The program is designed so that it will have massive parallelization on GPUs,
while still being more simple than standard Gaussian orbital based methods.

The code is based earlier Fortran code [DAGE](https://github.com/dagesundholm/DAGE) described in [chem.phys. 146, 084102 (2017)](http://dx.doi.org/10.1063/1.4976557).

This Julia implementation includes some upgrades for accuracy and advanced functionality like automatic differentation, while also being significantly faster and able to run on GPUs from multiple vendors.

## Development Plan

The aim is to run this code on [LUMI](https://www.lumi-supercomputer.eu/) supercomputer, which is operational in early 2022.

The plan is that the program is functional at LUMI lauch time, with DFT part done with pseudopotential support.

Later on we will add support for all electron calculations with [Bubbles](http://dx.doi.org/10.1021/acs.jctc.8b00456) framework.

## How Does it Work

The main idea here is to transform 3D Coulomb Integral to four one dimensional
integrals and thus masively reduce the computational time. This allows solving
Poisson equation. Helmholtz equation can then be solved using Greens function
the same way as Poisson equation, by only adding an extra constant term.

From the 4 one dimensinal integrals 3 can be calculated as a tensor contraction
that parallizes well on GPUs. The final integral can be parallized over different GPUs. This should result in a program that can parallize over tens of GPUs, for large systems.

THe main method also means that differential equation is solved as integral equation.
Thus the basis set is chosen to give the best accuracy for general numerical 
integrals, which means Gauss-Legedre polynomials. The system is also divided to
elements in order to reduse the maximum order of Gauss-Legedre polynomials.



## Highlighted Features (when complete)
- Solve Poisson equation (3D) - ready
- Solve Helmholtz equation (3D) - ready
- Solve Schrödinger equation with Helmholtz kernel Greens function - (1 particle ready)
- Parallelization on CPUs and GPUs - (CPU part working, GPU untested)
- Solve electronic structure with DFT - (todo)
- Calculate magnetic field efects on electronic structure - (1 particle ready)
- Full automatic differentation support - (forward mode working, reverse needs special pullbacks)

## Installation
Hit julia console "]" to enter pkg> and then type
```julia
using Pkg
pkg"add https://github.com/tjjarvinen/HKQM.jl"
```

## Code Examples

There are some examples in "examples" directory.

## Testing Accuracy

Simple test for accuracy can be performed by typing

```julia
using HKQM

a=5u"Å"   # system box size a^3
ne=4      # number of elements per dimension
ng=64     # number of Gauss points for r integration
nt=64     # number of Gauss points for t integration
tmax=700  # maximum t-value in integration

# Gaussian-Gaussian charge density repulsion
test_accuracy(a, ne, ng, nt; tmax=tmax)
```

[CI-img]: https://github.com/tjjarvinen/HKQM.jl/workflows/CI/badge.svg
[CI-url]: https://github.com/tjjarvinen/HKQM.jl/actions?query=workflow%3ACI

[codecov-img]: https://codecov.io/gh/tjjarvinen/HKQM.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/tjjarvinen/HKQM.jl
