



abstract type AbstractElement{Dims} end

abstract type AbstractElementArray{T,N} <: AbstractArray{T,N} end

abstract type AbstractElementGrid{N} <: AbstractArray{SVector{3,Float64}, N} end

"""
    Element1D <: AbstractElement{1}

Stores information for 1D element.

# Fields
- `low::Float64` : lowest value within element
- `high::Float64` : highest value within element
"""
struct Element1D <: AbstractElement{1}
    low::typeof(1.0u"bohr")
    high::typeof(1.0u"bohr")
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
    center::SVector{3,typeof(1.0u"bohr")}
    a::typeof(1.0u"bohr")
end

"""
    CubicElement(a) -> CubicElement

Return `CubicElement` with center at origin and side length `a`.
"""
function CubicElement(a)
    return CubicElement((0.,0.,0.).*u"bohr", a)
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
    CubicElements <: AbstractElementArray{CubicElement,3}

Cubic element that is divided to smaller cubic elments.

# Fields
- `a::typeof(1.0u"bohr")`                    : cubes side lenght
- `center::::SVector{3,typeof(1.0u"bohr")}`  : cube center
- `npoints`                                  : number of elements per degree of freedom
"""
struct CubicElementArray <: AbstractElementArray{CubicElement,3}
    a::typeof(1.0u"bohr")
    center::SVector{3,typeof(1.0u"bohr")}
    npoints::Int
    function CubicElementArray(a, npoints::Int; center=zeros(3).*u"bohr")
        @assert a > 0u"bohr"  "Element needs to have positive size"
        @assert npoints > 0 "Number of elements needs to be more than zero"
        new(a, center, npoints)
    end
end

const CubicElements = CubicElementArray

"""
    ElementVector <: AbstractElementArray{Element1D,1}

Holds several 1D elements that are next to each other.
In practice this just a array of `Element1D` with a condition that
they are next to each other.

# Fields
- `v::Vector{Element1D}`   :   Vector for elements

# Creation
    ElementVector(el::Element1D...)
    ElementVector(start, bounds...)

- `start`   :  lowest boundary value for elements
- `bounds`  :  boundaries for next elements, needs to have increasing values
"""
struct ElementVector <: AbstractElementArray{Element1D,1}
    v::Vector{Element1D}
    function ElementVector(el::Element1D...)
        if length(el) > 1
            s = el[begin].high
            for e in el[begin+1:end]
                @assert s == e.low  "Elements are not next to each other"
                s = e.high
            end
        end
        new(Vector(el))
    end
    function ElementVector(start, bounds...)
        s = start
        for b in bounds
            @assert s < b  "Invalid element boundaries"
            s = b
        end
        v = [Element1D(start, bounds[begin])]
        if length(bounds) > 1
            s = bounds[begin]
            for b in bounds[begin+1:end]
                push!(v, Element1D(s,b))
                s = b
            end
        end
        new(v)
    end
end


Base.size(ev::ElementVector) = size(ev.v)
Base.getindex(ev::ElementVector, i) = ev.v[i]


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



function get_1d_element(ce::CubicElements, i::Int)
    @assert 0 < i <= ce.npoints
    s = elementsize(ce)
    low = -0.5ce.a+(i-1)*s
    return Element1D(low, low+s)
end

function Base.getindex(ca::CubicElementArray, I::Int, J::Int, K::Int)
    low = ca.center .-  0.5 .* ca.a
    step = ca.a / ca.npoints
    c0 = low .+ (0.5.*step)
    nc = c0 .+ step .* ([I,J,K].-1)
    return CubicElement(nc, step)
end


function Base.show(io::IO, ce::CubicElements)
    print(io, "Cubic elements=$(size(ce))")
end

Base.size(ce::CubicElementArray) = (ce.npoints, ce.npoints, ce.npoints)

Base.show(io::IO, ce::CubicElement) = print(io, "Cubic element a = $(ce.a)")


"""
    get_center(ce::CubicElement)
    get_center(ce::CubicElementArray)

Return element center.
"""
get_center(ce::CubicElement) = ce.center
get_center(ce::CubicElementArray) = ce.center



## Element grids

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
julia> CubicElementGrid(5u"pm", 4, 32; origin=[1., 0., 0.].*u"nm)
Cubic elements grid with 4^3 elements with 32^3 Gauss points
```
"""
struct CubicElementGrid{NG, NE} <: AbstractElementGrid{6}
    elements::CubicElementArray
    ecenters::SVector{NE}{Float64}
    gpoints::SVector{NG}{Float64}
    w::SVector{NG}{Float64}
    origin::SVector{3}{Float64}
    function CubicElementGrid(a::Unitful.Length, nelements::Int, ngpoints::Int;
                              origin=SVector(0.,0.,0.).*unit(a))
        ce = CubicElementArray(a, nelements)
        x, w = gausspoints(ce, ngpoints)

        #TODO make this to take units properly
        ecenters = map( x-> austrip(get_center(x)[1]) , ce[:,1,1])
        new{ngpoints, nelements}(ce, ecenters, x, w, austrip.(origin))
    end
end

function Base.show(io::IO, ceg::CubicElementGrid{NG,NE}) where {NG,NE}
    print(io, "Cubic elements grid with $(NE)^3 elements with $(NG)^3 Gauss points")
end

function Base.show(io::IO, ::MIME"text/plain", ceg::CubicElementGrid{NG, NE}) where {NG,NE}
    print(io, "Cubic elements grid with $(NE)^3 elements with $(NG)^3 Gauss points")
end




function Base.size(c::CubicElementGrid)
    ng = length(c.gpoints)
    ne = size(c.elements)[1]
    return (ng, ng, ng, ne, ne, ne)
end


function Base.getindex(c::CubicElementGrid, i::Int,j::Int,k::Int, I::Int,J::Int,K::Int)
    return SVector(c.ecenters[I]+c.gpoints[i], c.ecenters[J]+c.gpoints[j], c.ecenters[K]+c.gpoints[k] )
end



"""
    xgrid(ceg::CubicElementGrid) -> Matrix{Float64}

x-axis coordinated of for elements and grids.
First index is for grids and second for elements.
"""
xgrid(ceg::CubicElementGrid) = [x+X+ceg.origin[1] for x in ceg.gpoints, X in ceg.ecenters ]

"""
    ygrid(ceg::CubicElementGrid) -> Matrix{Float64}

y-axis coordinated of for elements and grids.
First index is for grids and second for elements.
"""
ygrid(ceg::CubicElementGrid) = [x+X+ceg.origin[2] for x in ceg.gpoints, X in ceg.ecenters ]

"""
    zgrid(ceg::CubicElementGrid) -> Matrix{Float64}

z-axis coordinated of for elements and grids.
First index is for grids and second for elements.
"""
zgrid(ceg::CubicElementGrid) = [x+X+ceg.origin[3] for x in ceg.gpoints, X in ceg.ecenters ]

"""
    grid1d(ceg::CubicElementGrid) -> Matrix{Float64}

Return matrix of grid points in elements.
First index is for grids and second for elements.
Differs from [`xgrid`](@ref), [`ygrid`](@ref) and [`zgrid`](@ref) in that
the origin of grid is not specified. Which is equal to origin at 0,0,0.
"""
grid1d(ceg::CubicElementGrid) = [x+X for x in ceg.gpoints, X in ceg.ecenters ]





## Gauss points for integration


function gausspoints(n; elementsize=(-1.0, 1.0))
    x, w = gausslegendre(n)
    esize = austrip.(elementsize)
    shift = (esize[2]+esize[1])./2
    x = x .* (esize[2]-esize[1])./2 .+ shift
    w .*= (esize[2]-esize[1])/2
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
function gausspoints(ce::CubicElementArray, npoints::Int)
    s = elementsize(ce)/2
    return gausspoints(npoints; elementsize=(-s, s))
end
