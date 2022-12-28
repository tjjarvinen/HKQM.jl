

"""
    Element1D <: AbstractElement{1}

Stores information for 1D element.

# Fields
- `low::typeof(1.0u"bohr")` : lowest value within element
- `high::typeof(1.0u"bohr")` : highest value within element
"""
struct Element1D <: AbstractElement{1}
    low::typeof(1.0u"bohr")
    high::typeof(1.0u"bohr")
    function Element1D(low::Unitful.Length, high::Unitful.Length)
        @assert high > low
        new(low, high)
    end
end

function Element1D(low, high)
    @assert dimension(low) == dimension(high) == NoDims
    return Element1D(low*u"bohr", high*u"bohr")
end



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

Unitful.unit(e::Element1D) = unit(e.low)
Unitful.unit(ev::ElementVector) = unit(ev.v[begin])

"""
    element_size(e::Element1D)
    element_size(e::Element1D, T::DataType)

Return element length/size. `T` can be given to
define number type. Unit is given by the element.
"""
element_size(e::Element1D) = e.high - e.low

function element_size(e::Element1D, T::DataType)
    es = elemen_tsize(e)
    s = austrip(es) 
    c = convert(T, s)
    return c * unit(es)
end

"""
    getcenter(e::Element1D)

Center location of element
"""
getcenter(e::Element1D) = 0.5*(e.high + e.low)



## Element grids
# These have grid points in addition to elements


"""
    xgrid(egsb::AbstractElementGridSymmetricBox) -> Matrix

X-coordinates in grid from.
First index is for grids and second for elements.
"""
xgrid(egsb::AbstractElementGridSymmetricBox) = get_1d_grid(egsb)


"""
    ygrid(egsb::AbstractElementGridSymmetricBox) -> Matrix

Y-coordinates in grid from.
First index is for grids and second for elements.
"""
ygrid(egsb::AbstractElementGridSymmetricBox) = get_1d_grid(egsb)


"""
    zgrid(egsb::AbstractElementGridSymmetricBox) -> Matrix

Z-coordinates in grid from.
First index is for grids and second for elements.
"""
zgrid(egsb::AbstractElementGridSymmetricBox) = get_1d_grid(egsb)




## 1D grids
# Plan is to build noncubic grids based on these

struct ElementGrid{T} <: AbstractElementGrid{T, 1}
    basis::GaussLegendre{T}
    element::Element1D
    scaling::T
    shift::T
    function ElementGrid(element::Element1D, n::Int)
        scaling = ( element.high - element.low ) / 2 |> austrip
        shift = ( element.high + element.low ) / 2 |> austrip
        new{Float64}(GaussLegendre(n-1), element, scaling, shift)
    end
    function ElementGrid(DT::DataType, element::Element1D, n::Int)
        scaling = ( element.high - element.low ) / 2 |> austrip
        shift = ( element.high + element.low ) / 2 |> austrip
        new{DT}(GaussLegendre(n-1, DT), element, scaling, shift)
    end
end

ElementGrid(a, b, n) = ElementGrid(Element1D(a,b), n)
ElementGrid(T::DataType, a, b, n) = ElementGrid(T, Element1D(a,b), n)

Base.size(eg::ElementGrid) = size(eg.basis.nodes)
Base.getindex(eg::ElementGrid, i::Int) = muladd( eg.basis.nodes[i], eg.scaling, eg.shift )
Base.show(io::IO, ::ElementGrid) = print(io, "ElementGrid")


getweight(eg::ElementGrid) = eg.basis.weights .* eg.scaling
get_derivative_matrix(eg::ElementGrid) = eg.basis.D ./ eg.scaling
function element_size(eg::ElementGrid{T}) where {T}
    T(element_size(eg.element))
end 


function (eg::ElementGrid)(r, u)
    x = ( r .- eg.shift ) ./ eg.scaling
    return  interpolate(x, u, eg.basis)
end


struct ElementGridVector{T} <: AbstractElementGrid{T, 2}
    elements::Vector{ElementGrid{T}}
    function ElementGridVector(ev::ElementVector, ng)
        elements = [ ElementGrid(x, ng) for x in ev ]
        new{Float64}(elements)
    end
    function ElementGridVector(DT::DataType, ev::ElementVector, ng)
        elements = [ ElementGrid(DT, x, ng) for x in ev ]
        new{DT}(elements)
    end
end

function ElementGridVector(T::DataType, a, b, ne::Int, ng::Int)
    @assert a < b
    d = (b - a) / ne
    bounds = [ a + i*d for i in 1:ne ]
    ev = ElementVector(a, bounds...)
    return ElementGridVector(T, ev, ng)
end

ElementGridVector(a, b, ne::Int, ng::Int) = ElementGridVector(Float64, a, b, ne, ng)

Base.size(egv::ElementGridVector) = (length(egv.elements[1]), length(egv.elements))
Base.getindex(egv::ElementGridVector, i::Int, I::Int) = egv.elements[I][i]

getelement(egv::ElementGridVector, i::Int) = egv.elements[i].element

getweight(egv::ElementGridVector) = hcat( getweight.(egv.elements)... )
get_derivative_matrix(egv::ElementGridVector, i::Int=1) = get_derivative_matrix(egv.elements[i])
element_size(egv::ElementGridVector) = sum( element_size.(egv.elements)  )
element_size(egv::ElementGridVector, T::DataType) = sum( element_size.(egv.elements, T)  )

##

struct ElementGridSymmetricBox{T} <: AbstractElementGridSymmetricBox{T}
    egv::ElementGridVector{T}
end

function ElementGridSymmetricBox(a, ne::Int, ng::Int)
    egv = ElementGridVector(-a/2, a/2, ne, ng)
    return ElementGridSymmetricBox(egv)
end

function ElementGridSymmetricBox(T::DataType, a, ne::Int, ng::Int)
    egv = ElementGridVector(T, -a/2, a/2, ne, ng)
    return ElementGridSymmetricBox(egv)
end

function Base.size(egsb::ElementGridSymmetricBox)
    s = size(egsb.egv)
    return (s[1], s[1], s[1], s[2], s[2], s[2])
end

function Base.getindex(egsb::ElementGridSymmetricBox, i::Int,j::Int,k::Int, I::Int,J::Int,K::Int )
    return SVector( egsb.egv[i,I], egsb.egv[j,J], egsb.egv[k,K] )
end

function Base.show(io::IO, ::MIME"text/plain", egsb::ElementGridSymmetricBox)
    s = size(egsb)
    print(io, "ElementGridSymmetricBox $(s[end])^3 elements and $(s[1])^3 Gauss points per element")
end




getweight(egsb::ElementGridSymmetricBox) = getweight(egsb.egv)
get_derivative_matrix(egsb::ElementGridSymmetricBox, i::Int=1) = get_derivative_matrix(egsb.egv, i)
get_1d_grid(egsb::ElementGridSymmetricBox, i::Int=1) = egsb.egv  # symmetric thus drop i

function element_size(egsb::ElementGridSymmetricBox)
    a = element_size( egsb.egv )
    return a, a, a
end


##

function Base.minimum(ceg::AbstractElementGrid)
    return ceg[ firstindex(ceg) ] 
end

function Base.maximum(ceg::AbstractElementGrid)
    return ceg[ lastindex(ceg) ]
end

