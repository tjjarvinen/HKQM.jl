# This is meant to be the new elements that could be later taken as an own module
module HelmholtzKernel

using LinearAlgebra
using KernelAbstractions
using OhMyThreads: @tasks, @set, tmap
using PolynomialBases
using SpecialFunctions: erf, erfc
using StaticArrays
using TensorOperations
using Unitful
using UnitfulAtomic

export apply_helmholtz!
export apply_poisson!
export apply_transformation
export convert_variable_type
export default_transformation_tensor
export derivative_x
export derivative_x!
export derivative_y
export derivative_y!
export derivative_z
export derivative_z!
export element_bounds
export element_size
export get_center
export get_derivative_matrix
export get_element
export get_elementgrid
export get_weight
export integrate
export laplacian
export laplacian!

export AbstractElementGrid
export ConcreteTransformationTensor
export Element1D
export ElementGridArray
export ElementGridLegendre
export ElementGridLobatto
export ElementGridSymmetricBox
export ElementGridVectorLegendre
export ElementGridVectorLobatto
export ElementVector
export HelmholtzTensor
export PoissonTensor

struct KernelTensor1D{T}
    nodes::Vector{T}
    weights::Vector{T}
    Tmat::Array{T,3}   # Tmat[i, j, it]
end

struct KernelTensor3D{T}
    x::KernelTensor1D{T}
    y::KernelTensor1D{T}
    z::KernelTensor1D{T}
end


include("elements.jl")
include("greensfunctions.jl")
include("integrations.jl")
include("newkernels.jl")
include("utils.jl")
include("geensfunctions-new.jl")


end