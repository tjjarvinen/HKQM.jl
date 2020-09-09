using GaussQuadrature


abstract type AbstractElementCenters end

"""
    ElementCenters <: AbstractElementCenters

Struct to hold finite element center locations

# Fields
- `low`  : lowes value for element boundary
- `high` : highest value for element boundary
- `npoints` : number of elements per degree of freedom
"""
struct ElementCenters <: AbstractElementCenters
    low
    high
    npoints
    function ElementCenters(low, high, npoints::Int)
        @assert high > low
        @assert npoints > 0
        new(low, high, npoints)
    end
end



Base.size(ec::ElementCenters) = (ec.npoints, ec.npoints, ec.npoints)
function Base.getindex(ec::ElementCenters, i::Int)
    @assert i <= ec.npoints && i > 0
    s = (ec.high - ec.low)/ec.npoints
    return ec.low + (i-0.5)*s
end

function Base.getindex(ec::ElementCenters, i::Int, j::Int, k::Int)
    return ec[i], ec[j], ec[j]
end

function Base.show(io::IO, ec::ElementCenters)
    print(io, "Finite element centers size=$(size(ec))")
end

function elementsize(ec::ElementCenters)
    return (ec.high - ec.low)/ec.npoints
end

function getcenters(ec::ElementCenters)
    return [ec[i] for i ∈ 1:ec.npoints]
end

function gausspoints(n; elementsize=(-1.0, 1.0), shift=0.0)
    x, w = legendre(n)
    x = x .* (elementsize[2]-elementsize[1])./2 .+ shift
    return x, w
end

function gausspoints(ec::ElementCenters, npoints::Int)
    s = elementsize(ec)/2
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
