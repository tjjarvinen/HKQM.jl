# Toroidal and Poloidal Magnetic Currents

**This is a work in progress and does not give correct results at the moment!**

There is a submodule for toroidal-poloidal decomposition
of magnetic currents. To access it just

```julia
using TensorOperations
using HKQM
using HKQM.ToroidalCurrent
```

## SYSMOIC files

There are two options to read sysmoic files. A low level one

```julia
data = read_sysmoic(file_name)
```

and a high level one

```julia
J, dJ = read_current(file_name)
```

## Toroidal and Poloidal Currents

To calculate toroidal current type

```julia
J_toro = toroidal_current(dJ)
```

To calculte poloidal current type

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

## Writing results

You can write toroidal and poloidal current with

```julia
write_currents("file_name", J_toro, J_polo)

# change number of points per dimension
write_currents("file_name", J_toro, J_polo; n_points=25)
```

Distance is in Ångströms and order of variables is
x, y, z, Jx_toro, Jy_toro, Jz_toro, Jx_polo, Jy_polo, Jz_polo