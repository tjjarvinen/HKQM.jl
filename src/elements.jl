using GaussQuadrature
using FastGaussQuadrature


abstract type AbstractCubicElements end

"""
    CubicElements <: AbstractCubicElements

Struct to hold finite element center locations

# Fields
- `low`  : lowes value for element boundary
- `high` : highest value for element boundary
- `npoints` : number of elements per degree of freedom
"""
struct CubicElements <: AbstractCubicElements
    low
    high
    npoints
    function CubicElements(low, high, npoints::Int)
        @assert high > low
        @assert npoints > 0
        new(low, high, npoints)
    end
end



Base.size(ce::CubicElements) = (ce.npoints, ce.npoints, ce.npoints)
function Base.getindex(ce::CubicElements, i::Int)
    @assert i <= ce.npoints && i > 0
    s = (ce.high - ce.low)/ce.npoints
    return ce.low + (i-0.5)*s
end

function Base.getindex(ce::CubicElements, i::Int, j::Int, k::Int)
    return ce[i], ce[j], ce[j]
end

function Base.show(io::IO, ce::CubicElements)
    print(io, "Finite element centers size=$(size(ce))")
end

function elementsize(ce::CubicElements)
    return (ce.high - ce.low)/ce.npoints
end

function getcenters(ce::CubicElements)
    return [ce[i] for i ∈ 1:ce.npoints]
end

function gausspoints(n; elementsize=(-1.0, 1.0))
    x, w = legendre(n)
    shift = (elementsize[2]+elementsize[1])./2
    x = x .* (elementsize[2]-elementsize[1])./2 .+ shift
    w .*= (elementsize[2]-elementsize[1])/2
    return x, w
end

function gausspoints(ce::CubicElements, npoints::Int)
    s = elementsize(ce)/2
    return gausspoints(npoints; elementsize=(-s, s))
end

function gausspoints3d(x::AbstractVector, w::AbstractVector)
    xyz = collect(Base.Iterators.product(x,x,x))
    ww = zeros(n,n,n)
    for i ∈ 1:n
        for j ∈ 1:n
            for k ∈ 1:n
                ww[i,j,k] = w[i] * w[j] * w[k]
            end
        end
    end
    return xyz, ww
end

function gausspoints3d(n; elementsize=(-1.0, 1.0))
    return gausspoints3d( gausspoints(n; elementsize=elementsize) )
end
