
"""
CubicElement <: AbstractElement{3}

Cubic 3D element.

# Fields
- `center::SVector{3,Float64}`  : center location of element
- `a::typeof(1.0u"bohr")` : side length of the cube
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
    CubicElementArray <: AbstractElementArray{CubicElement,3}

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


"""
    elementsize(ce::CubicElementArray) -> Float64

Cubes 1D side length
"""
element_size(ce::CubicElementArray) = ce.a/ce.npoints



function get_1d_element(ce::CubicElementArray, i::Int)
    @assert 0 < i <= ce.npoints
    s = elementsize(ce)
    low = -0.5ce.a+(i-1)*s
    return Element1D(low, low+s)
end



function ElementVector(ce::CubicElementArray, i=1::Int)
    @assert i in 1:3
    start = -0.5*ce.a + ce.center[i]
    n = size(ce, i)
    shift = ce.a/ce.npoints
    b = [start + j*shift for j in 1:n]
    return ElementVector(start, b... )
end



function Base.getindex(ca::CubicElementArray, I::Int, J::Int, K::Int)
    low = ca.center .-  0.5 .* ca.a
    step = ca.a / ca.npoints
    c0 = low .+ (0.5.*step)
    nc = c0 .+ step .* ([I,J,K].-1)
    return CubicElement(nc, step)
end


function Base.show(io::IO, ce::CubicElementArray)
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




"""
    CubicElementGrid{NG, NE} <: AbstractElementGridSymmetricBox

Cube that is divided to smaller cubes that have integration grid.
Elements and integration points are symmetric for x-, y- and z-axes.

# Fields
- `elements::CubicElements` : cubic elements
- `ecenters::Vector{Float64}` : center locations for the subcube elements
- `gpoints::Vector{Float64}`  : integration grid definition, in 1D
- `w::Vector{Float64}`  :  integration weights
- `origin::SVector{3}{Float64}`  : origin location of the supercube

# Example
```jldoctest
julia> CubicElementGrid(5u"pm", 4, 32)
Cubic elements grid with 4^3 elements with 32^3 Gauss points
```
"""
struct CubicElementGrid{NG, NE} <: AbstractElementGridSymmetricBox{Float64}
    elements::CubicElementArray
    ecenters::Vector{Float64}
    gpoints::Vector{Float64}
    w::Vector{Float64}
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

function Base.show(io::IO, ceg::CubicElementGrid)
    print(io, "Cubic elements grid with $(size(ceg.elements,1))^3 elements with $(length(ceg.gpoints))^3 Gauss points")
end

function Base.show(io::IO, ::MIME"text/plain", ceg::CubicElementGrid)
    print(io, "Cubic elements grid with $(size(ceg.elements,1))^3 elements with $(length(ceg.gpoints))^3 Gauss points")
end




function Base.size(c::CubicElementGrid)
    ng = length(c.gpoints)
    ne = size(c.elements)[1]
    return (ng, ng, ng, ne, ne, ne)
end


function Base.getindex(c::CubicElementGrid, i::Int,j::Int,k::Int, I::Int,J::Int,K::Int)
    return SVector(c.ecenters[I]+c.gpoints[i], c.ecenters[J]+c.gpoints[j], c.ecenters[K]+c.gpoints[k] )
end


ElementVector(ceg::CubicElementGrid, i=1::Int) = ElementVector(ceg.elements, i)



xgrid(ceg::CubicElementGrid) = [x+X+ceg.origin[1] for x in ceg.gpoints, X in ceg.ecenters ]
ygrid(ceg::CubicElementGrid) = [x+X+ceg.origin[2] for x in ceg.gpoints, X in ceg.ecenters ]
zgrid(ceg::CubicElementGrid) = [x+X+ceg.origin[3] for x in ceg.gpoints, X in ceg.ecenters ]


"""
    grid1d(ceg::CubicElementGrid) -> Matrix{Float64}

Return matrix of grid points in elements.
First index is for grids and second for elements.
Differs from [`xgrid`](@ref), [`ygrid`](@ref) and [`zgrid`](@ref) in that
the origin of grid is not specified. Which is equal to origin at 0,0,0.
"""
grid1d(ceg::CubicElementGrid) = [x+X for x in ceg.gpoints, X in ceg.ecenters ]


function get_1d_grid(ceg::CubicElementGrid, i=1::Int)
    return ElementGridVector(ElementVector(ceg, i), size(ceg, i))
end

## Gauss points for integration




## Gauss points for integration


function gausspoints(n; elementsize=(-1.0, 1.0))
    x, w = gausslegendre(n)
    esize = austrip.(elementsize)
    shift = (esize[2]+esize[1])./2
    x = x .* (esize[2]-esize[1])./2 .+ shift
    w .*= (esize[2]-esize[1])/2
    return x, w
end


"""
    gausspoints(el::Element1D, n::Int) -> (SVector, SVector)

Create `n` Gauss-Legendre points for element.

Returns a tuple with first index having Gauss points and
second integration weights.
"""
function gausspoints(el::Element1D, n::Int)
    return gausspoints(n; elementsize=austrip.((el.low, el.high)) )
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
    s = element_size(ce)/2
    return gausspoints(npoints; elementsize=(-s, s))
end
