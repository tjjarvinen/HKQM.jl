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

Operators have unit defined acordinly [Unitful.jl](https://github.com/PainterQubits/Unitful.jl)

```@example guide
unit(r)
```

```@example guide
unit(p)
```

These operator are vector operator and have lenght defined

```@example guide
lenght(r)
```

Individual components can be accessed with indexing
```@example guide
x = r[1]
y = r[2]
z = r[3]
```

Operators support basic algebra operations

```@example guide
r + r
2 * r
r + [1u"bohr", 2u"Å", 1u"pm"]
r + 1u"bohr"    # add 1 bohr to all dimensions
```

Units are checked for the operations and operations that
do not make sense are prohibited
```@example guide
r + p
```

Vector operations are supported
```@example guide
r² = r ⋅ r   # = dot(r,r)
l  = r × p   # = cross(r,p)
```