



abstract type AbstractElement{Dims} end

abstract type AbstractElementGrid{N} <: AbstractArray{SVector{3,Float64}, N} end

"""
    Element1D <: AbstractElement{1}

Stores information for 1D element.

# Fields
- `low::Float64` : lowest value within element
- `high::Float64` : highest value within element
"""
struct Element1D <: AbstractElement{1}
    low::Float64
    high::Float64
    function Element1D(low, high)
        @assert high > low
        new(low, high)
    end
end

"""
    CubicElement <: AbstractElement{3}

Cubic 3D element.

# Fields
- `center::SVector{3,Float64}`  : center location of element
- `a::Float64` : side length of the cube
"""
struct CubicElement <: AbstractElement{3}
    center::SVector{3,Float64}
    a::Float64
end

"""
    CubicElement(a) -> CubicElement

Return `CubicElement` with center at origin and side length `a`.
"""
function CubicElement(a)
    return CubicElement((0.,0.,0.), a)
end

"""
    CubicElement(xe::Element1D, ye::Element1D, ze::Element1D) -> CubicElement

Construct `CubicElement` from three `Element1D`. Center is taken from elements,
which need to have same `elementsize`.
"""
function CubicElement(xe::Element1D, ye::Element1D, ze::Element1D)
    @assert elementsize(xe) == elementsize(ye) == elementsize(ze)
    center = SVector(getcenter(xe), getcenter(ye), getcenter(ze))
    return CubicElement(center, elementsize(xe))
end


"""
    CubicElements <: AbstractElement{3}

Cubic element that is divided to smaller cubic elments.

# Fields
- `a::Float64` : cubid side lenght
- `npoints` : number of elements per degree of freedom
"""
struct CubicElements <: AbstractElement{3}
    a::Float64
    npoints::Int
    function CubicElements(a, npoints::Int)
        @assert a > 0  "Element needs to have positive size"
        @assert npoints > 0 "Number of elements needs to be more than zero"
        new(a, npoints)
    end
end

"""
    CubicElementGrid{NG, NE} <: AbstractElementGrid{6}

Cube that is divided to smaller cubes that have integration grid.
Elements and integration points are symmetric for x-, y- and z-axes.

# Fields
- `elements::CubicElements` : cubic elements
- `ecenters::SVector{NE}{Float64}` : center locations for the subcube elements
- `gpoints::SVector{NG}{Float64}`  : integration grid definition, in 1D
- `w::SVector{NG}{Float64}`  :  integration weights
- `origin::SVector{3}{Float64}`  : origin location of the supercube

# Example
```jldoctest
julia> CubicElementGrid(5, 4, 32)  # 5 is in ångstöms
Cubic elements grid with 4^3 elements with 32^3 Gauss points

julia> using Unitful

julia> CubicElementGrid(5u"pm", 4, 32; origin=[1., 0., 0.].*u"nm)
Cubic elements grid with 4^3 elements with 32^3 Gauss points
```
"""
struct CubicElementGrid{NG, NE} <: AbstractElementGrid{6}
    elements::CubicElements
    ecenters::SVector{NE}{Float64}
    gpoints::SVector{NG}{Float64}
    w::SVector{NG}{Float64}
    origin::SVector{3}{Float64}
    function CubicElementGrid(a, nelements::Int, ngpoints::Int; origin=SVector(0.,0.,0.))
        ce = CubicElements(a, nelements)
        x, w = gausspoints(ce, ngpoints)
        new{ngpoints, nelements}(ce, getcenters(ce), x, w, origin)
    end
    function CubicElementGrid(a::Unitful.Quantity, nelements::Int, ngpoints::Int; origin=SVector(0.,0.,0.).*unit(a))
        @assert dimension(a) == dimension(u"m") "Dimension for \"a\" needs to be distance"
        ce = CubicElements(austrip(a), nelements)
        x, w = gausspoints(ce, ngpoints)
        new{ngpoints, nelements}(ce, getcenters(ce), x, w, austrip.(origin))
    end
end

function Base.show(io::IO, ceg::CubicElementGrid{NG,NE}) where {NG,NE}
    print(io, "Cubic elements grid with $(NE)^3 elements with $(NG)^3 Gauss points")
end

function Base.show(io::IO, ::MIME"text/plain", ceg::CubicElementGrid{NG, NE}) where {NG,NE}
    print(io, "Cubic elements grid with $(NE)^3 elements with $(NG)^3 Gauss points")
end

function Base.show(io::IO, ce::CubicElements)
    print(io, "Cubic elements=$(size(ce))")
end


Base.size(ce::CubicElements) = (ce.npoints, ce.npoints, ce.npoints)

function Base.size(c::CubicElementGrid{NG,NE}) where {NG, NE}
    return (NG, NG, NG, NE, NE, NE)
end


Base.length(ce::CubicElements) = ce.npoints

function Base.getindex(c::CubicElementGrid, i::Int,j::Int,k::Int, I::Int,J::Int,K::Int)
    return SVector(c.ecenters[I]+c.gpoints[i], c.ecenters[J]+c.gpoints[j], c.ecenters[K]+c.gpoints[k] )
end

function Base.getindex(ce::CubicElements, i::Int)
    @assert i <= ce.npoints && i > 0
    s = elementsize(ce)
    low = -0.5ce.a+(i-1)*s
    return Element1D(low, low+s)
end

function Base.getindex(ce::CubicElements, I, J, K)
    return CubicElement(ce[I], ce[J], ce[K])
end

"""
    elementsize(ce::CubicElements) -> Float64

Cubes 1D side length
"""
elementsize(ce::CubicElements) = ce.a/ce.npoints

"""
    elementsize(e::Element1D) -> Float64

Element length/size
"""
elementsize(e::Element1D) = e.high - e.low

"""
    getcenter(e::Element1D) -> Float64

Center location of element
"""
getcenter(e::Element1D) = 0.5*(e.high + e.low)

"""
    xgrid(ceg::CubicElementGrid) -> SMatrix

x-axis coordinated of for elements and grids.
First index is for grids and second for elements.
"""
xgrid(ceg::CubicElementGrid) = SMatrix([x+X+ceg.origin[1] for x in ceg.gpoints, X in ceg.ecenters ])

"""
    ygrid(ceg::CubicElementGrid) -> SMatrix

y-axis coordinated of for elements and grids.
First index is for grids and second for elements.
"""
ygrid(ceg::CubicElementGrid) = SMatrix([x+X+ceg.origin[2] for x in ceg.gpoints, X in ceg.ecenters ])

"""
    zgrid(ceg::CubicElementGrid) -> SMatrix

z-axis coordinated of for elements and grids.
First index is for grids and second for elements.
"""
zgrid(ceg::CubicElementGrid) = SMatrix([x+X+ceg.origin[3] for x in ceg.gpoints, X in ceg.ecenters ])

"""
    grid1d(ceg::CubicElementGrid) -> SMatrix

Return matrix of grid points in elements.
First index is for grids and second for elements.
Differs from [`xgrid`](@ref), [`ygrid`](@ref) and [`zgrid`](@ref) in that
the origin of grid is not specified. Which is equal to origin at 0,0,0.
"""
grid1d(ceg::CubicElementGrid) = SMatrix([x+X for x in ceg.gpoints, X in ceg.ecenters ])

"""
    getcenters(ce::CubicElements) -> Vector{Float64}

Return element centers.
"""
function getcenters(ce::CubicElements)
    return [ getcenter(ce[i]) for i ∈ 1:ce.npoints]
end

function gausspoints(n; elementsize=(-1.0, 1.0))
    x, w = gausslegendre(n)
    shift = (elementsize[2]+elementsize[1])./2
    x = x .* (elementsize[2]-elementsize[1])./2 .+ shift
    w .*= (elementsize[2]-elementsize[1])/2
    return SVector{n}(x), SVector{n}(w)
end


"""
    gausspoints(el::Element1D, n::Int) -> (SVector, SVector)

Create `n` Gauss-Legendre points for element.

Returns a tuple with first index having Gauss points and
second integration weights.
"""
function gausspoints(el::Element1D, n::Int)
    return gausspoints(n; elementsize=(el.low, el.high) )
end

"""
    gausspoints(ce::CubicElements, npoints::Int) -> (SVector, SVector)

Create `npoints` Gauss-Legendre points for element.

Returns a tuple with first index having Gauss points and
second integration weights.

Only one set of Gauss points that can be used for all elements is returned.
The returned points have center at 0 and need to be shifted for different elements.
"""
function gausspoints(ce::CubicElements, npoints::Int)
    s = elementsize(ce)/2
    return gausspoints(npoints; elementsize=(-s, s))
end
