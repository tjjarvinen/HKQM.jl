# CoulombIntegral
Solves electrostatic potential using combination finite elements and Gaussian quadrature.


## Installation
Hit "]" to enter pkg> and type
```julia
pkg> add https://github.com/tjjarvinen/CoulombIntegral.jl
```

## Testing Accuracy

Simple test can be performed by typing

```julia
julia> test_accuracy(16,16,64; tmax=25)
```

To see full instruction for test type

```julia
?test_accuracy
```

which results
```
search: test_accuracy

  test_accuracy(n_elements, n_gaussp, n_tpoints; tmax=10, correction=true, alt=false, cmax=10) -> Float

  Calculates self-energy for Gaussian change density numerically and compares it analytic result.

  Arguments
  ≡≡≡≡≡≡≡≡≡≡≡

    •    n_elements : Number of finite elements in calculation

    •    n_gaussp : Number of Gauss points in element per degree of freedom

    •    n_tpoins : Number of Gauss points for t-variable integration

  Keywords
  ≡≡≡≡≡≡≡≡≡≡

    •    tmax=20 : Maximum value in t integration

    •    correction=true : Add correction to electric field calculation

    •    alt=false : If true use alternative formulation for T-tensor, if false use original

    •    cmax=10 : Limits for numerical calculation - cubic box from -cmax to cmax
```

## Usage

Typing
```julia
using CoulombIntegral

E = self_energy(8,8,64)
```
will calculate self-interaction energy for benzene carbons.


To see what command options are available type
```julia
?self_energy
```
wich results
```
search: self_energy gaussiandensity_self_energy

  self_energy(n_elements, n_gaussp, n_tpoints; tmax=10, atoms=C6) -> Float64

  Calculates self energy using cubic elements and Gaussian quadrature.

  Arguments
  ≡≡≡≡≡≡≡≡≡≡≡

    •    n_elements : number of elements

    •    n_gaussp : number of Gauss points

    •    n_tpoints : number of Gauss points in exponential function integration

  Keywords
  ≡≡≡≡≡≡≡≡≡≡

    •    tmax=10 : integration limit for exponential function

    •    atoms=C6 : array of atom coordinates - default benzene carbons

    •    correction=true : add correction to t-coordinate integration
```
