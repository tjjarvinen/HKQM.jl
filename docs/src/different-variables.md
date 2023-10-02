# Changing Data Types and GPU Calculations

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

To change types use command `convert_variable_type`

```@example f32
eg_f64 = convert_variable_type(Float64, eg)

typeof(eg_f64)
```

```@example f32
qs = convert_variable_type(Float64, qs)

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

julia> typeof(qs_mtl)
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

## TensorOperations backend

The default backend maximizes compatibility to different
GPUs, but it is not as well optimized. To get a little bit
more performance you can use [TensorOperations](https://github.com/Jutho/TensorOperations.jl) backend, which also
supports forward mode AD.

To use TensorOperations backend, just load TensorOperations

```julia
using TensorOperations
using HKQM
```

you will also need to use `Array` type (or `CuArray`) to
use TensorOperations backend.

In the future, when TensorOperations will have other
backends, it will become the default backend (again).

## GPU Calculations

The main way to do GPU calculations is to change the array type to
the one GPU supports

 - `CuArray` -> [Nvidia GPUs](https://github.com/JuliaGPU/CUDA.jl)
 - `oneArray` -> [Intel GPUs](https://github.com/JuliaGPU/oneAPI.jl)
 - `ROCArray` -> [AMD GPUs](https://github.com/JuliaGPU/AMDGPU.jl)
 - `MtlArray` -> [Apple GPUs](https://github.com/JuliaGPU/Metal.jl)

For example to use on AMD GPU you could start by

```julia
using AMDGPU
using HKQM

eg = ElementGridSymmetricBox(Float64, 5u"Å", 4, 24)
qs = particle_in_box(ROCArray, eg, 1, 1, 1)
r = position_operator(ROCArray, eg)
p = momentum_operator(ROCArray, eg)

bracket(qs, r, qs)
bracket(qs, p, qs)
```

### Alternative TensorOperations backend for CUDA

For CUDA there is alternative TensorOperations backend
that from TensorOperations extension.

To use TensorOperations CUDA backend start by

```julia
using CUDA
using cuCUDA
using TensorOperations
using HKQM
```

and using `CuArray` type should now use TensorOperations cuTENSOR backend.
