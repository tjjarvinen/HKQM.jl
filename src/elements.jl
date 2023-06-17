

"""
    Element1D{T} <: AbstractElement{1}

Stores information for 1D element. Where `T<:Unitful.Length`.

# Fields
- `low::T` : lowest value within element
- `high::T` : highest value within element
"""
struct Element1D{T} <: AbstractElement{1}
    low::T
    high::T
    function Element1D(low::Unitful.Length, high::Unitful.Length)
        @assert high > low
        T = unit(low)
        tmp = promote( ustrip(low), ustrip(T, high) )
        new{ typeof(tmp[1] * T) }( tmp[1]*T, tmp[2]*T )
    end
end

function Element1D(low, high)
    #TODO remove this
    @assert dimension(low) == dimension(high) == NoDims
    return Element1D(low*u"bohr", high*u"bohr")
end

function Element1D(a)
    return Element1D(element_bounds(a)...)
end

Unitful.uconvert(T, e::Element1D) = Element1D( uconvert(T, e.low),  uconvert(T, e.high) )
Unitful.unit(e::Element1D) = unit(e.low)

element_bounds(e::Element1D) = (e.low, e.high)

"""
    element_size(e::Element1D)

Return element length/size.
"""
element_size(e::Element1D) = e.high - e.low


"""
    getcenter(e::Element1D)

Center location of element
"""
getcenter(e::Element1D) = (e.high + e.low) / 2


# Vector type to store elements of different sizes

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
        T = unit(el[begin])
        new( collect( uconvert.(T, el) ) )
    end
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
    ElementVector(v...)
end

Base.size(ev::ElementVector) = size(ev.v)
Base.getindex(ev::ElementVector, i) = ev.v[i]

Unitful.unit(ev::AbstractElementArray{<:Any,1}) = unit(ev[begin])
Unitful.uconvert(T, ev::ElementVector) = ElementVector( map( x->uconvert(T,x), ev)... )

function element_bounds(ev::AbstractElementArray{<:Any,1})
    low = element_bounds(ev[1])[1]
    high = element_bounds(ev[2])[2]
    return (low, high)
end

function element_size(ev::AbstractElementArray{<:Any,1})
    low, high = element_bounds(ev)
    return high - low
end


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

struct ElementGridLegendre{T} <: AbstractElementGrid{T, 1}
    basis::GaussLegendre{T}
    element::Element1D
    scaling::T
    shift::T
    function ElementGridLegendre(element, n::Int; unit=u"bohr")
        element = uconvert(unit, Element1D(element) )
        scaling = ustrip(unit, element_size(element) ) / 2 
        shift = ustrip(unit, sum(element_bounds(element)) ) / 2
        new{Float64}(GaussLegendre(n-1), element, scaling, shift)
    end
    function ElementGridLegendre(DT::DataType, element, n::Int; unit=u"bohr")
        element = uconvert(unit, Element1D(element) )
        scaling = ustrip(unit, element_size(element) ) / 2 
        shift = ustrip(unit, sum(element_bounds(element)) ) / 2
        new{DT}(GaussLegendre(n-1, DT), element, scaling, shift)
    end
end

struct ElementGridLobatto{T} <: AbstractElementGrid{T, 1}
    basis::LobattoLegendre{T}
    element::Element1D
    scaling::T
    shift::T
    function ElementGridLobatto(element, n::Int; unit=u"bohr")
        element = uconvert(unit, Element1D(element) )
        scaling = ustrip(unit, element_size(element) ) / 2 
        shift = ustrip(unit, sum(element_bounds(element)) ) / 2
        new{Float64}(LobattoLegendre(n-1), element, scaling, shift)
    end
    function ElementGridLobatto(DT::DataType, element, n::Int; unit=u"bohr")
        element = uconvert(unit, Element1D(element) )
        scaling = ustrip(unit, element_size(element) ) / 2 
        shift = ustrip(unit, sum(element_bounds(element)) ) / 2
        new{DT}(LobattoLegendre(n-1, DT), element, scaling, shift)
    end
end

const ElmentGrid = ElementGridLegendre

ElementGridLegendre(a, b, n) = ElementGridLegendre(Element1D(a,b), n)
ElementGridLobatto(a, b, n) = ElementGridLobatto(Element1D(a,b), n)
ElementGridLegendre(T::DataType, a, b, n) = ElementGridLegendre(T, Element1D(a,b), n)
ElementGridLobatto(T::DataType, a, b, n) = ElementGridLobatto(T, Element1D(a,b), n)

Base.size(eg::Union{ElementGridLegendre, ElementGridLobatto}) = size(eg.basis.nodes)
Base.getindex(eg::Union{ElementGridLegendre, ElementGridLobatto}, i::Int) = muladd( eg.basis.nodes[i], eg.scaling, eg.shift )
#Base.show(io::IO, ::ElementGridLegendre) = print(io, "ElementGrid")

Unitful.unit(eg::Union{ElementGridLegendre, ElementGridLobatto}) = unit(get_element(eg))

convert_variable_type(T, eg::ElementGridLegendre) = ElementGridLegendre(T, get_element(eg), length(eg))
convert_variable_type(T, eg::ElementGridLobatto) = ElementGridLobatto(T, get_element(eg), length(eg)) 

element_bounds(eg::Union{ElementGridLegendre, ElementGridLobatto}) = element_bounds( get_element(eg) )
function element_size(eg::AbstractElementGrid{<:Any, 1})
    low, high = element_bounds(eg)
    return high - low 
end

getweight(eg::Union{ElementGridLegendre, ElementGridLobatto}) = eg.basis.weights * eg.scaling
get_derivative_matrix(eg::Union{ElementGridLegendre, ElementGridLobatto}) = eg.basis.D / eg.scaling
function element_size(eg::Union{ElementGridLegendre{T}, ElementGridLobatto{T}}) where {T}
    T(element_size(eg.element))
end 

get_element(eg::Union{ElementGridLegendre, ElementGridLobatto}) = eg.element


function (eg::Union{ElementGridLegendre, ElementGridLobatto})(r, u)
    x = @. ( r - eg.shift ) / eg.scaling
    return interpolate(x, u, eg.basis)
end


struct ElementGridVector{T} <: AbstractElementGrid{T, 2}
    elements::Vector{ElementGridLegendre{T}}
    function ElementGridVector(ev::ElementVector, ng)
        elements = [ ElementGridLegendre(x, ng) for x in ev ]
        new{Float64}(elements)
    end
    function ElementGridVector(DT::DataType, ev::ElementVector, ng)
        elements = [ ElementGridLegendre(DT, x, ng) for x in ev ]
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

function convert_variable_type(T, egv::ElementGridVector)
    a = egv.elements[begin].element.low
    b = egv.elements[end].element.high
    ne = size(egv, 2)
    ng = size(egv, 1)
    return ElementGridVector(T, a, b, ne, ng)
end

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

function convert_variable_type(T, egsb::ElementGridSymmetricBox)
    return ElementGridSymmetricBox( convert_variable_type(T, egsb.egv) )
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




getweight(egsb::ElementGridSymmetricBox, i::Int=1) = (getweight ∘ get_1d_grid)(egsb, i)
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

