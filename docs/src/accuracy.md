# Accuracy Testing

There are several ways on how to test accuracy of the calculations,
which is nessary in order to know how good the results are.

## Accuracy of Poisson equation Greens function

To test accuracy of Poisson equation and Helmholtz equation
use `test_accuracy` function

```@docs
test_accuracy
```

It can be called with

```@example accuracy
test_accuracy(5u"Å", 4, 24)
```

Which gives accuracy for 5 Å box with 4 elements per dimension and 24^3 Gauss-Lagrange points per element.

The default mode uses `optimal_coulomb_tranformation` to calculate Poisson equation

```docs
optimal_coulomb_tranformation
```

This is usually good enough.

### Integration tuning

!!! note "Work in progress"

  This is work in progress. More coming later...

To perform integration there is a so called `t` variable that has to be integrated from zero to infinity.
The main contribution comes from small values.

- Normal mode will use normal Gauss-Lagrange itegration
- Logarithmic will distribute the points in logarithmic fashion
- Local mean that average value around the points is calculated. This can be used with normal and logarithmic spacing

Fanally there is correction for very large values.

To test the integral accuracy calculate without correction and choose how many
points are used for `t`-integrarion `nt`.

In example to use logarithmic spacing to integrate from `t=20` to `t=70` one
can use

```@example
test_accuracy(5u"Å", 4, 24, 24; mode=:log, tmin=20, tmax=70, correction=false)
```

From the output we can see that integral is heavily overestimated,
see "integration error" from output.

To test the same aree with `loglocal` mode gives

```@example
test_accuracy(5u"Å", 4, 24, 24; mode=:loglocal, tmin=20, tmax=70, correction=false)
```

and we can see that the accuracy was considerably improved.

To build integral tensor from parts you can use `AbstractHelmholtzTensor` types

```docs
HelmholtzTensorLinear
HelmholtzTensorLog
HelmholtzTensorLocalLinear
HelmholtzTensorLocalLog
HelmholtzTensorCombination
```

To overload the default tensor for calculations you need to create (=redefine) `optimal_coulomb_tranformation`
function that returns `HelmholtzTensor`.
