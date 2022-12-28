# Changing Data Types

## Changing Variable Type

Standard calculations use `Float64`. This can be changed to vide variety of options.

```@example f32
using HKQM

eg = ElementGridSymmetricBox(Float32, 5u"Å", 4, 24)
typeof(eg)
```

This will persist to operators

```@example f32
r = position_operator(eg)

typeof(r[1])
```

and quantum states

```@example f32
qs = particle_in_box(eg, 1, 1, 1)

typeof(qs)
```

### Changing Type of Existing Data

To change types use command `change_variable_type`

```@example f32
eg_f64 = change_variable_type(Float64, eg)

typeof(eg_f64)
```

```@example f32
qs = change_variable_type(Float64, qs)

typeof(qs)
```

## Changing Array Type

You can change array type quantum states and operators.'
This can be done on construct time
(here using [Metal.jl](https://github.com/JuliaGPU/Metal.jl))

```julia
julia> using Metal

julia> qs_mtl = particle_in_box(MtlArray, eg, 1, 1, 1)
Quantum state

julia> typeof(q_mtl)
QuantumState{MtlArray{Float32, 6}, Float32}

julia> r_mtl = position_operator(MtlArray, eg)
Operator 4^3 elements, 24^3 Gauss points per element

julia> typeof(r_mtl[1])
ScalarOperator{MtlArray{Float32, 6}}
```

or by converting existing structures using `convert_array_type`

```julia
julia> qs_new = convert_array_type(MtlArray, qs)
Quantum state

julia> typeof(qs_new)
QuantumState{MtlArray{Float32, 6}, Float32}

julia> r_new = convert_array_type(MtlArray, r)
Operator 4^3 elements, 24^3 Gauss points per element

julia> typeof(r_new[1])
ScalarOperator{MtlArray{Float32, 6}}
```
