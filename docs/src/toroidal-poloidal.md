# Toroidal and Poloidal Magnetic Currents

**This is a work in progress and does not give correct results at the moment!**

## Definition

*Toroidal* current (ψ) is defined as

```math
-\Delta_2 \psi = \frac{\partial J_y}{\partial x} - \frac{\partial J_x}{\partial y}
```

*Poloidal* current (ϕ) is defined as

```math
-\Delta_2 \phi = J_z
```

Where

```math
\Delta_2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}
```

Green's function that solves the system is 2D Poisson kernel

```math
G(x_1, y_1; x_0, y_0) = \frac{1}{4\pi} \ln (\Delta x^2 + \Delta y^2)
```

## Submodule

There is a submodule for toroidal-poloidal decomposition
of magnetic currents. To access it just

```julia
using HKQM
using HKQM.ToroidalCurrent
```

## SYSMOIC files

There are two options to read Sysmoic files. A low level one

```julia
data = read_sysmoic(file_name)
```

and a high level one

```julia
J, dJ = read_current(file_name)
```

You can control the precision of integration grid

```julia
ne = 4  # Number of elements per dimension
ng = 32 # Number of Gauss points per element per dimension
J, dJ = read_current(file_name, ne, ng)
```

This affects memory usage. So, if you have problems with memory, try to lower precision.

## Toroidal and Poloidal Currents

To calculate toroidal current type

```julia
J_toro = toroidal_current(dJ)
```

To calculate poloidal current type

```julia
J_polo = poloidal_current(J)
```

To check that everything is correct type

```julia
# This should be zero current
ΔJ = J[:,3] - ( J_toro + J_polo )

# If everything is correct this returns zeros
bracket.(ΔJ, ΔJ)
```

### Adjust integration parameters

Poisson kernel includes a calculation of logarithm of zero distance, which is minus infinity.
To counter this, there is a small additional number added to the logarithm

```math
\ln (\Delta x^2 + \Delta y^2) \approx \ln (\Delta x^2 + \Delta y^2 + \epsilon)
```

This will make sure no logarithm is taken from zero.

You can change this variable by calling

```julia

# 1.0 * 10^-7 Å change in distance
J_toro = toroidal_current(dJ; eps=1E-7u"Å")

# 1.0 * 10^-5 bohr change in distance
J_polo = poloidal_current(J; eps=1E-5u"bohr")
```

## Writing results

You can write toroidal and poloidal current with

```julia
write_currents("file_name", J_toro, J_polo)

# change number of points per dimension
write_currents("file_name", J_toro, J_polo; n_points=25)
```

Distance is in Ångströms and order of variables is
x, y, z, Jx_toro, Jy_toro, Jz_toro, Jx_polo, Jy_polo, Jz_polo
