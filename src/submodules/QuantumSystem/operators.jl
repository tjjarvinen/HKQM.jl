abstract type AbstractOperator{N} end

get_elementgrid(ao::AbstractOperator) = ao.elementgrid
Base.size(ao::AbstractOperator) = size(get_elementgrid(ao))
Base.length(::AbstractOperator{N}) where N = N
Base.firstindex(::AbstractOperator) = 1
Base.lastindex(ao::AbstractOperator) = length(ao)

function Base.show(io::IO, ao::AbstractOperator)
    s = size(ao)
    print(io, "Operator with grid size $s")
end

function Base.getindex(so::AbstractOperator{1}, i::Int)
    @assert i == 1
    return so
end

function Base.iterate(v::AbstractOperator, i=0)
    if i < length(v)
        return (v[i+1], i+1)
    else
        return nothing
    end
end

Unitful.unit(ao::AbstractOperator) = ao.unit
Unitful.dimension(ao::AbstractOperator) = dimension(unit(ao))

function Unitful.uconvert(u::Unitful.Units, op::AbstractOperator)
    @assert dimension(u) == dimension(op)
    unit(op) == u && return op
    T = (eltype ∘ eltype)(get_elementgrid(op)) # make sure type is correct
    conv = (T ∘ ustrip ∘ uconvert)(u, 1*unit(op)) * u / unit(op)
    return conv*op
end

Base.:(+)(ao::AbstractOperator) = ao
Base.:(-)(ao::AbstractOperator) = -1*ao

dot(v1::AbstractOperator{N}, v2::AbstractOperator{N}) where N = sum(v1.*v2)

function cross(v1::AbstractOperator{3}, v2::AbstractOperator{3})
    r1 = v1[2]*v2[3] - v1[3]*v2[2]
    r2 = v1[1]*v2[3] - v1[3]*v2[1]
    r3 = v1[1]*v2[2] - v1[2]*v2[1]
    return VectorOperator(r1,r2,r3)
end

function cross(v1::AbstractOperator{3}, v2::AbstractVector)
    @assert length(v2) == 3
    r1 = v1[2]*v2[3] - v1[3]*v2[2]
    r2 = v1[1]*v2[3] - v1[3]*v2[1]
    r3 = v1[1]*v2[2] - v1[2]*v2[1]
    return [r1,r2,r3]
end

Base.:(*)(op::AbstractOperator{1}, qs::QuantumState) = op(qs)
Base.:(*)(op::AbstractOperator, qs::QuantumState) = map(f->f(qs), op)



## ScalarOperator

struct ScalarOperator{TA, T, D} <: AbstractOperator{1} where{TA<:AbstractArray{T, D}, T, D<:Int}
    elementgrid::AbstractElementGrid
    vals::TA
    unit::Unitful.FreeUnits
    function ScalarOperator(ega::AbstractElementGrid{<:Any, D}, vals::AbstractArray{T, D}; unit=NoUnits) where {T,D}
        new{typeof(vals), eltype(vals), D}(ega, vals, unit)
    end
end

get_values(so::ScalarOperator) = so.vals

function convert_array_type(T, so::ScalarOperator)
    return ScalarOperator(get_elementgrid(so), T(so.vals); unit=unit(so) )
end

function (so::ScalarOperator{<:Any,<:Any,D})(qs::QuantumState{<:Any,<:Any,D}) where D
    @assert size(so) == size(qs)
    return QuantumState(get_elementgrid(so), so.vals.*qs.psi, unit(so)*unit(qs))
end


function Base.map(f, so::ScalarOperator; output_unit=unit(so))
    return ScalarOperator(
        get_elementgrid(so),
        f.(so.vals);
        unit=output_unit
    )    
end


function Base.map(
    f, 
    so1::ScalarOperator{<:Any, <:Any, D}, 
    so2::ScalarOperator{<:Any, <:Any, D}; 
    output_unit=unit(so1)
) where D
    @assert size(so1) == size(so2)
    return ScalarOperator(
        get_elementgrid(so1),
        f.(so1.vals, so2.vals);
        unit=output_unit
    )    
end


function Base.:(+)(
    so1::ScalarOperator{<:Any, T1, D}, 
    so2::ScalarOperator{<:Any, T2, D}
) where {T1, T2, D}
    @assert size(so1) == size(so2)
    @assert dimension(so1) == dimension(so2)
    T = promote_type(T1, T2)
    conv = ustrip(unit(so1), one(T)*unit(so2))
    return map(so1, so2) do a,b
        muladd(conv, b, a)
    end
end

function Base.:(-)(
    so1::ScalarOperator{<:Any, T1, D}, 
    so2::ScalarOperator{<:Any, T2, D}
) where {T1, T2, D}
    @assert size(so1) == size(so2)
    @assert dimension(so1) == dimension(so2)
    T = promote_type(T1, T2)
    conv = -ustrip(unit(so1), one(T)*unit(so2))
    return map(so1, so2) do a,b
        muladd(conv, b, a)
    end
end

for OP in (:(+), :(-))
    @eval begin
        function Base.$OP(a::ScalarOperator, x::Number)
            @assert dimension(a) == dimension(x) "Dimension missmatch"
            y = ustrip(uconvert(unit(a), x))
            return map( b->$OP(b,y), a)
        end
        function Base.$OP(x::Number, a::ScalarOperator)
            @assert dimension(a) == dimension(x) "Dimension missmatch"
            y = ustrip(uconvert(unit(a), x))
            return map( b-> $OP(y,b), a)
        end
    end
end


for OP in (:(/), :(*))
    @eval begin
        function Base.$OP(a::ScalarOperator,b::ScalarOperator)
            return map( (x,y)-> $OP(x,y), a, b; output_unit=$OP(unit(a), unit(b)))
        end
        function Base.$OP(a::ScalarOperator, x::Number)
            return map( b-> $OP(b,ustrip(x)), a; output_unit=$OP(unit(a), unit(x)) )
        end
        function Base.$OP(x::Number, a::ScalarOperator)
            return map( b-> $OP(ustrip(x),b), a; output_unit=$OP(unit(x), unit(a)) )
        end
    end
end


# Functions that need unitless input
for op in (:sin, :cos, :tan, :exp, :log)
    @eval function Base.$op(so::ScalarOperator)
        @assert dimension(so) == dimension(NoUnits) "Operator needs to be dimensionless"
        return map($op, so)
     end
end

# Functions that can change units
for op in (:sqrt, :cbrt)
    @eval Base.$op(so::ScalarOperator) = map($op, so; output_unit=$op(unit(so)))
end

# Functions that dont change units
for op in (:real, :imag)
    @eval Base.$op(so::ScalarOperator) = map($op, so; output_unit=unit(so))
end

Base.:(-)(a::ScalarOperator) = -1*a
Base.:(^)(a::ScalarOperator, x::Number) = map( y->y^x, a; output_unit=unit(a)^x)

function Unitful.uconvert(a::Unitful.Units, so::ScalarOperator{<:Any, T, <:Any}) where T
    @assert dimension(a) == dimension(so) "Dimension missmatch"
    a == unit(so) && return so
    conv = (T ∘ ustrip ∘ uconvert)(u, 1*unit(so))
    return map( x->conv*x, so; output_unit=u)
end


