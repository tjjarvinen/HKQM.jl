abstract type AbstractElement{Dims} end
abstract type AbstractElementArray{T,N} <: AbstractArray{T,N} end
abstract type AbstractElementGrid{T,N} <: AbstractArray{T,N} end



## 1D element. All else a based on this
"""
    Element1D{T} <: AbstractElement{1}

Stores information for 1D element.

# Fields
- `low::T` : lowest value within element
- `high::T` : highest value within element
"""
struct Element1D{T} <: AbstractElement{1}
    low::T
    high::T
    function Element1D(low, high)
        @assert high > low
        @assert dimension(low) == dimension(high)
        T = unit(low)
        tmp = promote( ustrip(low), ustrip(T, high) )
        new{ typeof(tmp[1] * T) }( tmp[1]*T, tmp[2]*T )
    end
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
    get_center(e::Element1D)

Center location of element
"""
get_center(e::Element1D) = (e.high + e.low) / 2


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

function ElementVector(l::Unitful.Length, n::Integer)
    d = l/n
    bounds = [ i*d - l/2   for i in 1:n ]
    return ElementVector(-l/2, bounds...)
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




## Elementgrids. These can be used as basis

struct ElementGridLegendre{T} <: AbstractElementGrid{T, 1}
    basis::GaussLegendre{T}
    element::Element1D
    scaling::T
    shift::T
    function ElementGridLegendre(DT::DataType, element, n::Int)
        scaling = ustrip(element_size(element) ) / 2 
        shift = ustrip(sum(element_bounds(element)) ) / 2
        new{DT}(GaussLegendre(n-1, DT), element, scaling, shift)
    end
end

struct ElementGridLobatto{T} <: AbstractElementGrid{T, 1}
    basis::LobattoLegendre{T}
    element::Element1D
    scaling::T
    shift::T
    function ElementGridLobatto(DT::DataType, element, n::Int)
        scaling = (ustrip ∘ element_size)(element) / 2 
        shift = (ustrip ∘ sum ∘ element_bounds)(element) / 2
        new{DT}(LobattoLegendre(n-1, DT), element, scaling, shift)
    end
end


ElementGridLegendre(element, n::Int) = ElementGridLegendre(Float64, element, n)
ElementGridLobatto(element, n::Int) = ElementGridLobatto(Float64, element, n)
ElementGridLegendre(a, b, n) = ElementGridLegendre(Element1D(a,b), n)
ElementGridLobatto(a, b, n) = ElementGridLobatto(Element1D(a,b), n)
ElementGridLegendre(T::DataType, a, b, n) = ElementGridLegendre(T, Element1D(a,b), n)
ElementGridLobatto(T::DataType, a, b, n) = ElementGridLobatto(T, Element1D(a,b), n)

Base.getindex(eg::Union{ElementGridLegendre, ElementGridLobatto}, i::Int) = muladd( eg.basis.nodes[i], eg.scaling, eg.shift )
Base.show(io::IO, ::ElementGridLegendre) = print(io, "ElementGridLegendre")
Base.show(io::IO, ::ElementGridLobatto) = print(io, "ElementGridLobatto")
Base.size(eg::Union{ElementGridLegendre, ElementGridLobatto}) = size(eg.basis.nodes)

Unitful.unit(eg::Union{ElementGridLegendre, ElementGridLobatto}) = unit(get_element(eg))

convert_variable_type(T, eg::ElementGridLegendre) = ElementGridLegendre(T, get_element(eg), length(eg))
convert_variable_type(T, eg::ElementGridLobatto) = ElementGridLobatto(T, get_element(eg), length(eg)) 

element_bounds(eg::Union{ElementGridLegendre, ElementGridLobatto}) = element_bounds( get_element(eg) )

function element_size(eg::AbstractElementGrid{<:Any, 1})
    low, high = element_bounds(eg)
    return high - low 
end

get_derivative_matrix(eg::Union{ElementGridLegendre, ElementGridLobatto}) = eg.basis.D / eg.scaling
get_weight(eg::Union{ElementGridLegendre, ElementGridLobatto}) = eg.basis.weights * eg.scaling

function element_size(eg::Union{ElementGridLegendre{T}, ElementGridLobatto{T}}) where {T}
    T(element_size(eg.element))
end 

get_element(eg::Union{ElementGridLegendre, ElementGridLobatto}) = eg.element

# Used to interpolate. Need to calculate some values u and then value(s) at r can be calculated
function (eg::Union{ElementGridLegendre, ElementGridLobatto})(r, u)
    x = @. ( r - eg.shift ) / eg.scaling
    return interpolate(x, u, eg.basis)
end




## These combine several elements. They are the main building blocks for basis

struct ElementGridVectorLegendre{T} <: AbstractElementGrid{T, 1}
    elements::Vector{ElementGridLegendre{T}}
    index::Vector{Pair{Int,Int}}
    function ElementGridVectorLegendre(eg::ElementGridLegendre...)
        if length(eg) > 1
            s = element_bounds(eg[begin])[2]
            u = unit(eg[begin])
            for i in 2:length(eg)
                tmp = element_bounds(eg[i])
                @assert s ≈ tmp[1] "Elements need to be next to each other"
                s = tmp[2]
                @assert u == unit(eg[i]) "Elements need to have same unit"
            end
        end
        tmp = Pair{Int,Int}[]
        for ie in 1:length(eg)
            for ig in 1:length(eg[ie])
                push!(tmp, ie=>ig)
            end
        end
        new{eltype(eg[begin])}(collect(eg), tmp)
    end
end

struct ElementGridVectorLobatto{T} <: AbstractElementGrid{T, 1}
    elements::Vector{ElementGridLobatto{T}}
    index::Vector{Pair{Int,Int}}
    function ElementGridVectorLobatto(eg::ElementGridLobatto...)
        if length(eg) > 1
            s = element_bounds(eg[begin])[2]
            u = unit(eg[begin])
            for i in 2:length(eg)
                tmp = element_bounds(eg[i])
                @assert s ≈ tmp[1] "Elements need to be next to each other"
                s = tmp[2]
                @assert u == unit(eg[i]) "Elements need to have same unit"
            end
        end
        tmp = Pair{Int,Int}[]
        for ie in 1:length(eg)
            for ig in 1:length(eg[ie]) - 1 # skip last point
                push!(tmp, ie=>ig)
            end
        end
        # Add last point
        l = length(eg)
        ll = length(eg[l])
        push!(tmp, l=>ll)
        new{eltype(eg[begin])}(collect(eg), tmp)
    end
end

function ElementGridVectorLobatto(ev::ElementVector, ng)
    elements = [ ElementGridLobatto(x, ng) for x in ev ]
    return ElementGridVectorLobatto(elements...)
end

function ElementGridVectorLobatto(DT::DataType, ev::ElementVector, ng)
    elements = [ ElementGridLobatto(DT, x, ng) for x in ev ]
    return ElementGridVectorLobatto(elements...)
end

function ElementGridVectorLobatto(l::Unitful.Length, ne, ng)
    ev = ElementVector(l, ne)
    return ElementGridVectorLobatto(ev, ng)
end


function ElementGridVectorLegendre(ev::ElementVector, ng)
    elements = [ ElementGridLegendre(x, ng) for x in ev ]
    return ElementGridVectorLegendre(elements...)
end

function ElementGridVectorLegendre(DT::DataType, ev::ElementVector, ng)
    elements = [ ElementGridLegendre(DT, x, ng) for x in ev ]
    return ElementGridVectorLegendre(elements...)
end

function ElementGridVectorLegendre(l::Unitful.Length, ne, ng)
    ev = ElementVector(l, ne)
    return ElementGridVectorLegendre(ev, ng)
end


Base.size(egv::Union{ElementGridVectorLegendre, ElementGridVectorLobatto}) = size(egv.index)

function Base.getindex(egv::Union{ElementGridVectorLegendre, ElementGridVectorLobatto}, i::Int)
    return egv.elements[ egv.index[i].first ][egv.index[i].second]
end

Unitful.unit(egv::Union{ElementGridVectorLegendre, ElementGridVectorLobatto}) = unit(egv.elements[begin])

function convert_variable_type(T, egv::ElementGridVectorLegendre)
    return ElementGridVectorLegendre( convert_variable_type.(T, egv.elements)... )
end

function convert_variable_type(T, egv::ElementGridVectorLobatto)
    return ElementGridVectorLobatto( convert_variable_type.(T, egv.elements)... )
end

function element_bounds(egv::Union{ElementGridVectorLegendre, ElementGridVectorLobatto}) 
    low = element_bounds( egv.elements[begin])[1]
    high = element_bounds( egv.elements[end])[2]
    return (low, high)
end

function get_derivative_matrix(egv::ElementGridVectorLegendre)
    D = zeros(eltype(egv), length(egv), length(egv))
    i = 1
    for el in egv.elements
        l = i + length(el) - 1
        tmp = get_derivative_matrix(el)
        D[i:l,i:l] = tmp
        i = l+1
    end
    return D
end

function get_derivative_matrix(egv::ElementGridVectorLobatto)
    if length(egv.elements) == 1
        return get_derivative_matrix(egv.elements[begin])
    end
    # Build derivative matrix and integration weights
    dv = [ derivative_matrix(x) for x in egv.elements ]
    cr = [ length(x) for x in egv.elements ]
    # Size of derivative matrix. Lobatto basis overlaps with adjacent elements
    # and we need to remove this overlap, by reducing number of points.
    l = length(egv)
    D = zeros(eltype(egv) ,l, l)  # Derivate matrix
    D[1:cr[1], 1:cr[1]] = dv[1]
    # Index range for final derivative matrix build from element ones
    tmp = ones(Int, length(cr))
    tmp[1] = 0
    cum_index = cumsum( cr .- tmp )
    for i in 2:length(cum_index)
        ir = cum_index[i-1]:cum_index[i]
        D[ir, ir] += dv[i]
        D[cum_index[i-1],:] .*= 0.5 # remove double count from overlap point
    end
    return D
end


function get_weight(egv::ElementGridVectorLegendre)
    return reduce( vcat,  map( x->get_weight(x), egv.elements ) )
end

function get_weight(egv::ElementGridVectorLobatto)
    if length(egv.elements) == 1
        return get_weight(egv.elements[1])
    end
    cr = [length(x) for x in egv.elements]
    l = length(egv)
    w = zeros(eltype(egv), l)      # Integration weights
    w[1:cr[1]] = get_weight(egv.elements[1])
    # Index range for final derivative matrix build from element ones
    tmp = ones(Int, length(cr))
    tmp[1] = 0
    cum_index = cumsum( cr .- tmp )
    for i in 2:length(cum_index)
        ir = cum_index[i-1]:cum_index[i]
        w[ir] += get_weight(egv.elements[i])
    end
    return w
end


## Multidimensional element grids

struct ElementGridArray{T,TV,TA,N} <: AbstractElementGrid{SVector{N,T}, N}
    elements::Vector{AbstractElementGrid{T, 1}}
    derivatives::Vector{TA}
    r::Vector{TV}
    weights::Vector{TV}
    function ElementGridArray(egv::AbstractElementGrid{T, 1}...; array_type=Array) where T
        d = [ array_type( get_derivative_matrix(x) ) for x in egv]
        w = [ array_type( get_weight(x) ) for x in egv]
        r = [ array_type( x ) for x in egv ]
        new{T, typeof(w[begin]), typeof(d[begin]), length(egv)}(collect(egv), d, r, w)
    end
end


function Base.size(ega::ElementGridArray)
    return Tuple( length(x) for x in ega.r )
end

@generated function Base.getindex(ega::ElementGridArray{T,<:Any,<:Any,N}, i::Int...) where {N,T}
    if N != length(i)
        error("Not supported to call with $(length(i)) indices, you need to call with $N indices")
    end
    s = "SVector{$N,$T}("
    for j in 1:N
        s *= "ega.r[$j][(i[$j])],"
    end
    s *= ")"
    return Meta.parse(s)
end

function Base.show(io::IO, ::MIME"text/plain", ega::ElementGridArray)
    s = size(ega)
    print(io, "ElementGridArray of size = $(s)")
end

function Base.show(io::IO, ega::ElementGridArray)
    s = size(ega)
    print(io, "ElementGridArray of size = $(s)")
end

get_elementgrid(ega::ElementGridArray, i::Integer) = ega.elements[i]
get_derivative_matrix(ega::ElementGridArray, i::Integer) = ega.derivatives[i]
get_weight(ega::ElementGridArray, i::Integer) = ega.weights[i]

element_bounds(ega::ElementGridArray, i::Integer) = element_bounds(ega.elements[i])
element_size(ega::ElementGridArray, i::Integer) = element_size(ega.elements[i])

##

function get_derivative_matrix(grid, index::Integer, order::Integer)
    if order == 1
        return get_derivative_matrix(grid, index)
    else
        return get_derivative_matrix(grid, index)^order
    end
end