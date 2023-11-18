# This is meant to be the new elements that could be later taken as an own module
module HelmholtzKernel

using PolynomialBases
using StaticArrays
using Unitful
using UnitfulAtomic

export element_bounds
export element_size
export get_center

include("elements.jl")
include("integrations.jl")




end