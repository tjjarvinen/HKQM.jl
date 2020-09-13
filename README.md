# CoulombIntegral
Solves electrostatic potential using combination finite elements and Gaussian quadrature.

**Note integration weights are not correct yet!**

## Installation
Hit "]" to enter pkg> and type
```julia
pkg> add https://github.com/tjjarvinen/CoulombIntegral.jl
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
search: self_energy

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
```
