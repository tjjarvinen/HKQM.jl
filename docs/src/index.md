# HKQM.jl

```@meta
DocTestSetup = quote
    using HKQM
end
```

Documentation for HKQM.jl

## Installation

Installed with Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPEL more and run
```
pkg> add https://github.com/tjjarvinen/HKQM.jl
```

To test install type
```
pkg> test HKQM
```


## Running with CPU

With CPU there are two things that define how many threads are
used Julia number of threads and BLAS number of threads. This option affects on how many cores tensor contractions are done.

From these Julia number of threads can be defined on startup with `-t` option. It can be checked once started with command

```julia
Base.Threads.nthreads()
```

The second option is how many threads BLAS is using. This can be find out with command

```julia
using LinearAlgebra
LinearAlgebra.BLAS.get_num_threads()
```

Setting up number of threads for BLAS is done with

```julia
LinearAlgebra.BLAS.set_num_threads(n)
```

### Number of Processes

Number of Processes defines how many instances of Poisson/Helmholtz Greens functions are run on parallel. That is how many orbitals are updated in parallel. You do not want this option to be higher than number of orbitals. Ideally number of orbitals can be divided by number or processes.

When done on same computer Julia can be started with option `-p`
that defines number of processes used in calculation. See [documentation](https://docs.julialang.org/en/v1/manual/distributed-computing/#Starting-and-managing-worker-processes) for details. Alternatively you can use Distributed package to start more processes

```julia
using Distributed

addprocs(n)
```

Number of worker processes can found out by typing

```julia
nworkers()
```

## Package Extension

Package extensions are in place for

- [TensorOperations](https://github.com/Jutho/TensorOperations.jl) - alternative framework for contractions
- [Tullio](https://github.com/mcabbott/Tullio.jl) - extra type of nuclear potentials
- [TensorOperations](https://github.com/Jutho/TensorOperations.jl) with [CUDA](https://github.com/JuliaGPU/CUDA.jl) - TensorOperations CUDA backend
