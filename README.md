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

test_accuracy(16,16,64; tmax=25)
```

To see full instruction for test type

```julia
?test_accuracy
```

which results
```
help?> test_accuracy
search: test_accuracy

  test_accuracy(n_elements, n_gaussp, n_tpoints; Keywords) -> Float

  Calculates self-energy for Gaussian change density numerically and compares it analytic result.

  Arguments
  ≡≡≡≡≡≡≡≡≡≡≡

    •    n_elements : Number of finite elements in calculation

    •    n_gaussp : Number of Gauss points in element per degree of freedom

    •    n_tpoins : Number of Gauss points for t-variable integration

  Keywords
  ≡≡≡≡≡≡≡≡≡≡

    •    tmax=20 : Maximum value in t integration

    •    tmin=0 : Minimum value in t integration

    •    correction=true : Add correction to electric field calculation

    •    cmax=10 : Limits for numerical calculation - cubic box from -cmax to cmax

    •    mode=:normal : Integration method :normal, :normal_alt or :harrison

    •    δ=1.0 : Area parameter to determine average value in :normal_alt δ∈]0,1]

    •    μ=0 : Parameter for :harrison mode
```
