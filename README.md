# CoulombIntegral
Solves electrostatic potential using combination finite elements and Gaussian quadrature.


## Installation
Hit julia console "]" to enter pkg> and then type
```julia
pkg> add https://github.com/tjjarvinen/CoulombIntegral.jl
```
Alternatively type
```julia
using Pkg
Pkg.add(url="https://github.com/tjjarvinen/CoulombIntegral.jl")
```

## Testing Accuracy

Simple test can be performed by typing

```julia
using CoulombIntegral
a=10   # system box size a^3
ne=4   # number of elements
ng=64  # number of Gauss points for r
nt=64  # number of Gauss points for t

test_accuracy(a, ne, ng, nt; tmax=300, mode=:combination)
```
