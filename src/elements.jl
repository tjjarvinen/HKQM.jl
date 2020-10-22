using GaussQuadrature
using StaticArrays


abstract type AbstractElement{Dims} end

abstract type AbstractElementGrid{N} <: AbstractArray{Tuple{Float64,Float64,Float64}, N} end


struct Element1D <: AbstractElement{1}
    low::Float64
    high::Float64
    function Element1D(low, high)
        @assert high > low
        new(low, high)
    end
end

struct CubicElement <: AbstractElement{3}
    center::SVector{3,Float64}
    a::Float64
end

function CubicElement(xe::Element1D, ye::Element1D, ze::Element1D)
    @assert elementsize(xe) == elementsize(ye) == elementsize(ze)
    center = SVector(getcenter(xe), getcenter(ye), getcenter(ze))
    return CubicElement(center, elementsize(xe))
end


"""
    CubicElements <: AbstractCubicElements

Struct to hold finite element center locations

# Fields
- `low`  : lowes value for element boundary
- `high` : highest value for element boundary
- `npoints` : number of elements per degree of freedom
"""
struct CubicElements
    low::Float64
    high::Float64
    npoints::Int
    function CubicElements(low, high, npoints::Int)
        @assert high > low
        @assert npoints > 0
        new(low, high, npoints)
    end
end



struct CubicElementGrid{NG, NE} <: AbstractElementGrid{6} where {NG}
    elements::CubicElements
    ngpoints::Int
    ecenters::SVector{NE}{Float64}
    gpoints::SVector{NG}{Float64}
    w::SVector{NG}{Float64}
    function CubicElementGrid(elow, ehigh, nelements, ngpoints)
        ce = CubicElements(elow, ehigh, nelements)
        x, w = gausspoints(ce, ngpoints)
        new{ngpoints, nelements}(ce, ngpoints, getcenters(ce), x, w)
    end
end

function Base.show(io::IO, ceg::CubicElementGrid)
    print(io, "Cubic elements grid with $(size(ceg.elements)[1])^3 elements and $(ceg.ngpoints)^3 Gauss points")
end

function Base.show(io::IO, ::MIME"text/plain", ceg::CubicElementGrid)
    print(io, "Cubic elements grid with $(size(ceg.elements)[1])^3 elements and $(ceg.ngpoints)^3 Gauss points")
end

function Base.show(io::IO, ce::CubicElements)
    print(io, "Cubit elements=$(size(ce))")
end


Base.size(ce::CubicElements) = (ce.npoints, ce.npoints, ce.npoints)

function Base.size(c::CubicElementGrid)
    return ((c.ngpoints, c.ngpoints, c.ngpoints)..., size(c.elements)...)
end


Base.length(ce::CubicElements) = ce.npoints

function Base.getindex(c::CubicElementGrid, i::Int,j::Int,k::Int, I::Int,J::Int,K::Int)
    return (c.ecenters[I]+c.gpoints[i], c.ecenters[J]+c.gpoints[j], c.ecenters[K]+c.gpoints[k] )
end

function Base.getindex(ce::CubicElements, i::Int)
    @assert i <= ce.npoints && i > 0
    s = elementsize(ce)
    low = ce.low+(i-1)*s
    return Element1D(low, low+s)
end

function Base.getindex(ce::CubicElements, I, J, K)
    return CubicElement(ce[I], ce[J], ce[K])
end


elementsize(ce::CubicElements) = (ce.high - ce.low)/ce.npoints
elementsize(e::Element1D) = e.high - e.low

getcenter(e::Element1D) = 0.5*(e.high + e.low)

get1dgrid(ceg::CubicElementGrid) = SMatrix([x+X for x in ceg.gpoints, X in ceg.ecenters ])

function getcenters(ce::CubicElements)
    return [ getcenter(ce[i]) for i ∈ 1:ce.npoints]
end

function gausspoints(n; elementsize=(-1.0, 1.0))
    x, w = legendre(n)
    shift = (elementsize[2]+elementsize[1])./2
    x = x .* (elementsize[2]-elementsize[1])./2 .+ shift
    w .*= (elementsize[2]-elementsize[1])/2
    return SVector{n}(x), SVector{n}(w)
end

function gausspoints(el::Element1D, n)
    return gausspoints(n; elementsize=(el.low, el.high) )
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
