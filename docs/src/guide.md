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

The grid is also an Array that can be used as one

```@example guide
typeof(ceg) <: AbstractArray
```

```@example guide
size(ceg)
```

```@example guide
eltype(ceg)
```

The values are x-, y- and z-coordinates of the grid point in bohr.

```@example guide
ceg[1,1,1,3,3,3]
```

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
V = -30u"eV" * exp(-0.1u"bohr^-2" * r²) 

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
ψ = QuantumState( exp(-0.2u"bohr^-2" * r²) )
normalize!(ψ)

bracket(ψ, H, ψ)
```

After that Helmholtz Greens function can be used to generate better estimate for the lowest eigen state

```@example guide
ϕ, E = solve_eigen_states(H, ψ; max_iter=10, rtol=1E-6)
```

You can add more states to the solution by giving more intial states

```@example guide
ψ111 = particle_in_box(ceg, 1,1,1)
ψ112 = particle_in_box(ceg, 1,1,2)

ϕ2, E2 = solve_eigen_states(H, ψ111, ψ112)
```

Once estimate is self consistent a true solution has been found.

## Solving Hartree-Fock equation

Hartree-Fock equation can be solver with `scf` command.

```@example guide
V = -100u"eV" * exp(-0.1u"bohr^-2" * r²) 
H = HamiltonOperator(V)

ψ₁ = QuantumState( exp(-0.2u"bohr^-2" * r²) )
ψ₂ = 1u"bohr^-1"*r[1]*QuantumState( exp(-0.2u"bohr^-2" * r²) )
sd = SlaterDeterminant(ψ₁, ψ₂)
```

To check that all eigen values are negative calculate Fock matrix and look for diagonal values.

```julia
fock_matrix(sd, H)
```

After that you can solve Hartree-Fock equations

```julia
sd1 = scf(sd, H; tol=1E-6, max_iter=10)
```

`tol` is maximum chance in orbital overlap untill convergence is
archieved. `max_iter` is maximum iterations calculated.

Hartree-Fock energy is calculated by calling `hf_energy`

```julia
hf_energy(sd1, H)
```

Orbital energies can be found from diagonal of Fock matrix

```julia
fock_matrix(sd1, H)
```

Check also that offdiagonal elements are insignificant to make sure
the system real solution has been found.

## Approximate Nuclear Potential

There is an approximation to nuclear potential defined in here
[J. Chem. Phys. 121, 11587 (2004)](https://doi.org/10.1063/1.1791051).
It allows approximate electronic structure calculations.

Here is an example for Hydrogen molecule.

Define nuclear positions

```@example guide
r₁ = [0.37, 0., 0.] .* 1u"Å"
r₂ = [-0.37, 0., 0.] .* 1u"Å"
```

After that create nuclear potential

```@example guide
V₁ = nuclear_potential_harrison_approximation(ceg, r₁, "H")
V₂ = nuclear_potential_harrison_approximation(ceg, r₂, "H")

V = V₁ + V₂
```

and Hamiltonian

```@example guide
H = HamiltonOperator(V)
```

Create initial orbital

```@example guide
ϕ = particle_in_box(ceg, 1,1,1)
ψ = SlaterDeterminant( ϕ )
```

Solve SCF equations

```julia
ψ1 = scf(ψ, H)
```
