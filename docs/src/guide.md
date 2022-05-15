# Package Guide

## Define Grid for Calculations

Start by defining grid used in calculations

```@example guide
using HKQM

ceg = ElementGridSymmetricBox(5u"Å", 4, 24)
```

This creates a cubic box with with side lenght of 5 Å that are
divided to 4 elements and 24 Gauss-Lagrange points for each
element. Resulting in total of `(4*24)^3=884736` points.

## Operator algebra

Generate basic operators for the grid

```@example guide
r = position_operator(ceg)
p = momentum_operator(ceg)
```

Operators have units defined with [Unitful.jl](https://github.com/PainterQubits/Unitful.jl) package

```@example guide
unit(r)
```

```@example guide
unit(p)
```

These operator are vector operator and have lenght defined

```@example guide
length(r)
```

Individual components can be accessed with indexing

```@example guide
x = r[1]
y = r[2]
z = r[3]

nothing # hide
```

Operators support basic algebra operations

```@example guide
r + r
2 * r
r + [1u"bohr", 2u"Å", 1u"pm"]
r + 1u"bohr"
r / 2
x * y

nothing # hide
```

Units are checked for the operations and operations that
do not make sense are prohibited

```julia
r + p
```

Vector operations are supported

```@example guide
r² = r ⋅ r
l  = r × p

nothing # hide
```

Common functions can be used also

```@example guide
sin(1u"bohr^-1" * x)
exp(-1u"bohr^-2" * r²)

nothing # hide
```

Functions require that the input is unitless

```julia
exp(-r²)
```

## Quantum States

Quantum states can be created from operators

```@example guide
ψ = QuantumState( exp(-2u"bohr^-2" * r²) )
```

States can be normalized

```@example guide
normalize!(ψ) 
```

Complex conjugate can be taken with

```@example guide
ϕ = conj(ψ)
conj!(ψ)

nothing # hide
```

Quantum states have linear algebra defined

```@example guide
2ψ - ψ
```

Inner product can be calculated with `bracket` function

```@example guide
bracket(ψ, 2ψ) 
```

Operators can be applied to quantum state by multiplication

```@example guide
x * ψ
```

Vector operators return arrays of quantum state

```@example guide
r * ψ
```

Quantum states have units

```@example guide
unit(ψ) 
```

```@example guide
unit( x * ψ ) 
```

Other [Unitful](https://github.com/PainterQubits/Unitful.jl) functions like
`dimension` and `uconvert` are defined also.

Expectational values of operators can be calculated with `bracket` funtion

```@example guide
bracket(ψ, x, ψ) 
```

## Slater Determinant

Slater determinant is orthonormal set of quantum states

```@example guide
st = SlaterDeterminant([ψ, (1u"bohr^-1"*x)*ψ])
```

Slater determinat is an array of orbitals represented by quantum states

```@example guide
length(st)
```

```@example guide
st[1]
```

## Hamilton Operator

Hamilton operator is a special operator that is needed for Helmholtz Greens function.
To create it you need to create potential energy operator first

```@example guide
V = -20u"eV" * exp( exp(-0.25u"bohr^-2" * r²) )

H = HamiltonOperator(V)
```

Default mass is one electron mass and it can be customised with `m` keyword

```@example guide
H_2me = HamiltonOperator(V; m=2u"me_au")

nothing # hide
```

## Solving Eigen States of a Hamiltonian

You need to generate initial state for Hamiltonian that gives negative energy!

```@example guide
ψ = QuantumState( exp(-1u"bohr^-2" * r²) )
normalize!(ψ)

bracket(ψ, H, ψ)
```

After that Helmholtz Greens function can be used to generate better estimate for the lowest eigen state

```@example guide
ϕ = helmholtz_equation(ψ, H)
```

Update to estimate can be done in place too

```@example guide
helmholtz_equation!(ϕ, H)
```

Once estimate is self consistent a true solution has been found.
