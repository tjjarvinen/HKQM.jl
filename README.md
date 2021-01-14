# HKQM (Helmholtz Kernel Quantum Mechanics)
This package is under development and not ready for use yet!

The main idea here is to transform 3D Coulomb Integral to four one dimensional
integrals and thus masively reduce the computational time. This allows solving
Poisson equation. Helmholtz equation can then be solved using Greens function
the same way as Poisson equation.

Development is done mainly to implement quantum chemical program that has massive
parallelization on GPUs. But the methodology allows use in other fields too, like
quantum mechanics in general or any field that needs to solve Helmholtz or Poisson
equations in 3D. If you are integrested in using on other fields please be free to
contact us.

## Higlighted Features (when complete)
- Solve Poisson equation (3D) - ready
- Solve Helmholtz equation (3D) - ready
- Solve Schrödinger equation with Helmholtz kernel Greens function - (1 particle ready)
- Parallelization on CPUs and GPUs - (CPU part working, GPU untested)
- Solve electronic structure with DFT - (todo)
- Calculate magnetic field efects on electronic structure - (1 particle ready)
- Full AD support - (forward mode working, reverse needs special pullbacks)

## Installation
Hit julia console "]" to enter pkg> and then type
```julia
using Pkg
pkg"add https://github.com/tjjarvinen/HKQM.jl"
```

## Testing Accuracy

Simple test can be performed by typing

```julia
using HKQM

a=10   # system box size a^3 (in bohr)
ne=4   # number of elements
ng=64  # number of Gauss points for r
nt=64  # number of Gauss points for t
tmax=700 # maximum t-value in integration

test_accuracy(a, ne, ng, nt; tmax=tmax)
```
