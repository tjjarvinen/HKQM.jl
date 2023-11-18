# This is meant to be the new elements that could be later taken as an own module
module HelmholtzKernel

using LinearAlgebra: mul!
using PolynomialBases
using StaticArrays
using Unitful
using UnitfulAtomic

export convert_variable_type
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
export get_weight
export integrate
export laplacian
export laplacian!

export Element1D
export ElementGridArray
export ElementGridLegendre
export ElementGridLobatto
export ElementGridVectorLegendre
export ElementGridVectorLobatto
export ElementVector

include("elements.jl")
include("integrations.jl")




end