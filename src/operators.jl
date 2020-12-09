import LinearAlgebra.dot
import LinearAlgebra.cross
using Unitful
using UnitfulAtomic


abstract type AbstractOperator end

abstract type AbstractScalarOperator <: AbstractOperator end
abstract type AbstractVectorOperator <: AbstractOperator end
abstract type AbstractCompositeOperator <: AbstractOperator end

Base.size(ao::AbstractOperator) = size(ao.elementgrid)
Base.show(io::IO, ao::AbstractOperator) = print(io, "Operator size=$(size(ao))")
Unitful.unit(ao::AbstractOperator) = ao.unit
Unitful.dimension(ao::AbstractOperator) = dimension(unit(ao))

## Scalar Operator

struct ScalarOperator{T} <: AbstractScalarOperator
    elementgrid::CubicElementGrid
    vals::T
    unit::Unitful.FreeUnits
    function ScalarOperator(ceg::CubicElementGrid, ψ::AbstractArray{<:Any,6}; unit::Unitful.FreeUnits=NoUnits)
        new{typeof(ψ)}(ceg, ψ, unit)
    end
end

(so::ScalarOperator)(ψ::AbstractArray{<:Any,6}) = QuantumState(so.elementgrid, so.vals.*ψ)


for OP in (:(+), :(-))
    @eval begin
        function Base.$OP(a::ScalarOperator,b::ScalarOperator)
            @assert dimension(a) == dimension(b) "Dimension missmatch"
            c = uconvert(unit(a), b)
            ScalarOperator(a.elementgrid, $OP.(a.vals,c.vals); unit=a.unit)
        end
        function Base.$OP(a::ScalarOperator, x::Number)
            @assert dimension(a) == dimension(x) "Dimension missmatch"
            y = uconvert(unit(a), x)
            return ScalarOperator(a.elementgrid, $OP.(a.vals,y); unit=a.unit)
        end
        function Base.$OP(x::Number, a::ScalarOperator)
            @assert dimension(a) == dimension(x) "Dimension missmatch"
            y = uconvert(unit(a), x)
            return ScalarOperator(a.elementgrid, $OP.(y,a.vals); unit=a.unit)
        end
    end
end

for OP in (:(/), :(*))
    @eval begin
        function Base.$OP(a::ScalarOperator,b::ScalarOperator)
            ScalarOperator(a.elementgrid, $OP.(a.vals,b.vals); unit=$OP(unit(a), unit(b)))
        end
        function Base.$OP(a::ScalarOperator, x::Number)
            return ScalarOperator(a.elementgrid, $OP.(a.vals,x); unit=$OP(unit(a), unit(x)))
        end
        function Base.$OP(x::Number, a::ScalarOperator)
            return ScalarOperator(a.elementgrid, $OP.(x,a.vals); unit=$OP(unit(x), unit(a)))
        end
    end
end

Base.:(^)(a::ScalarOperator, x::Number) = ScalarOperator(a.elementgrid, (^).(a.vals,x); unit=unit(a)^x)


for op in (:sin, :cos, :exp, :log, :sqrt, :cbrt, :real, :imag)
    @eval Base.$op(so::ScalarOperator) = ScalarOperator(so.elementgrid, $op.(so.vals); unit=$op(so.unit))
end

function Unitful.uconvert(a::Unitful.Units, so::ScalarOperator)
    @assert dimension(a) == dimension(so) "Dimension missmatch"
    a == unit(so) && return so
    conv = ustrip(uconvert(a, 1*unit(so)))
    return ScalarOperator(so.elementgrid, so.vals.*conv; unit=a)
end


## Vector Operator

struct VectorOperator{N} <: AbstractVectorOperator where N
    elementgrid::CubicElementGrid
    operators::Vector{AbstractScalarOperator}
    function VectorOperator(op::AbstractScalarOperator...)
        si = size.(op)
        @assert all(map(x-> x==si[begin], si))
        new{length(op)}(op[1].elementgrid, [op...])
    end
    function VectorOperator(op::AbstractVector{<:AbstractScalarOperator})
        si = size.(op)
        @assert all(map(x-> x==si[begin], si))
        new{length(op)}(op[1].elementgrid, op)
    end
end

Base.firstindex(::VectorOperator) = 1
Base.getindex(v::VectorOperator, i...) = v.operators[i...]
Base.lastindex(v::VectorOperator) = length(v)
Base.length(v::VectorOperator) = length(v.operators)

Unitful.unit(v::VectorOperator) = unit(v[begin])

for OP in (:(+), :(-))
    @eval begin
        function Base.$OP(a::VectorOperator{N},b::VectorOperator{N}) where N
            VectorOperator($OP(a.operators,b.operators))
        end
    end
end

for OP in (:(+), :(-), :(*))
    @eval begin
        Base.$OP(a::VectorOperator, x::Number) = VectorOperator($OP.(a.operators,x))
        Base.$OP(x::Number, a::VectorOperator) = VectorOperator($OP.(x,a.operators))

        function Base.$OP(v::VectorOperator,s::ScalarOperator)
            @assert size(v) == size(s)
            VectorOperator( [$OP(x,s) for x in v.operators] )
        end
        function Base.$OP(s::ScalarOperator,v::VectorOperator)
            @assert size(v) == size(s)
            VectorOperator( [$OP(s,x) for x in v.operators] )
        end
    end
end


for OP in [:(/)]
    @eval begin
        function Base.$OP(v::VectorOperator,s::ScalarOperator)
            @assert size(v) == size(s)
            VectorOperator( [$OP(x,s) for x in v.operators] )
        end
        Base.$OP(a::VectorOperator, x::Number) = VectorOperator($OP.(a.operators,x))
    end
end


function Base.iterate(v::VectorOperator, i=0)
    if i < length(v)
        return (v[i+1], i+1)
    else
        return nothing
    end
end

dot(v1::VectorOperator, v2::VectorOperator) = sum(v1.*v2)

function cross(v1::VectorOperator, v2::VectorOperator)
    r1 = v1[2]*v2[3] - v1[3]*v2[2]
    r2 = v1[1]*v2[3] - v1[3]*v2[1]
    r3 = v1[1]*v2[2] - v1[2]*v2[1]
    return VectorOperator(r1,r2,r3)
end

## Position operator

function PositionOperator(ceg::CubicElementGrid)
    x(v) = v[1]
    y(v) = v[2]
    z(v) = v[3]
    return VectorOperator(
                          ScalarOperator(ceg, x.(ceg); unit=u"a0_au"),
                          ScalarOperator(ceg, y.(ceg); unit=u"a0_au"),
                          ScalarOperator(ceg, z.(ceg); unit=u"a0_au")
                         )
end


## Derivative Operator

struct DerivativeOperator{NG,N} <: AbstractScalarOperator
    elementgrid::CubicElementGrid
    dt::DerivativeTensor{NG}
    function DerivativeOperator(ceg::CubicElementGrid, coordinatate=1)
        @assert coordinatate ∈ 1:3 "coordinate needs to be 1, 2 or 3"
        dt = DerivativeTensor(ceg)
        new{size(ceg)[1], coordinatate}(ceg, dt)
    end
end


function (d::DerivativeOperator{<:Any,1})(ψ::AbstractArray{<:Any,6})
    return QuantumState(d.elementgrid, operate_x(d.dt, ψ) )
end

function (d::DerivativeOperator{<:Any,2})(ψ::AbstractArray{<:Any,6})
    return QuantumState(d.elementgrid, operate_y(d.dt, ψ) )
end

function (d::DerivativeOperator{<:Any,3})(ψ::AbstractArray{<:Any,6})
    return QuantumState(d.elementgrid, operate_z(d.dt, ψ) )
end

Unitful.unit(::DerivativeOperator) = u"1/a0_au"

## Gradient Operator
struct GradientOperator{NG} <: AbstractVectorOperator
    elementgrid::CubicElementGrid
    dt::DerivativeTensor{NG}
    function GradientOperator(ceg::CubicElementGrid)
        dt = DerivativeTensor(ceg)
        new{size(ceg)[1]}(ceg, dt)
    end
end


function (go::GradientOperator)(ψ::AbstractArray{<:Any,6})
    return [QuantumState(go.elementgrid, operate_x(go.dt,ψ)),
            QuantumState(go.elementgrid, operate_y(go.dt,ψ)),
            QuantumState(go.elementgrid, operate_z(go.dt,ψ))]
end

function (go::GradientOperator)(ψ::QuantumState)
    return go(ψ.ψ)
end

function (go::GradientOperator)(a::AbstractVector{<:AbstractQuantumState})
    @assert length(a) == 3
    @assert all(map(x->size(x)==size(go), a))
    out = operate_x(go.dt, a[1].ψ)
    out .+= operate_y(go.dt, a[2].ψ)
    out .+= operate_z(go.dt, a[3].ψ)
    return QuantumState(go.ceg, out)
end

#(go::GradientOperator)(ψ::QuantumState) = go(ψ.ψ)

Base.show(io::IO, go::GradientOperator) = print(io::IO, "Gradient operator size=$(size(go))")


## derivative operations
function operate_x!(dψ::AbstractArray{<:Any,6}, dt::DerivativeTensor, ψ::AbstractArray{<:Any,6})
    tmp = similar(ψ, size(dt)...)
    tmp .= dt
    @tensor dψ[i,j,k,I,J,K] = tmp[i,l] * ψ[l,j,k,I,J,K]
    return dψ
end

function operate_y!(dψ::AbstractArray{<:Any,6}, dt::DerivativeTensor, ψ::AbstractArray{<:Any,6})
    tmp = similar(ψ, size(dt)...)
    tmp .= dt
    @tensor dψ[i,j,k,I,J,K] = tmp[j,l] * ψ[i,l,k,I,J,K]
    return dψ
end

function operate_z!(dψ::AbstractArray{<:Any,6}, dt::DerivativeTensor, ψ::AbstractArray{<:Any,6})
    tmp = similar(ψ, size(dt)...)
    tmp .= dt
    @tensor dψ[i,j,k,I,J,K] = tmp[k,l] * ψ[i,j,l,I,J,K]
    return dψ
end


function operate_x(dt::DerivativeTensor, ψ::AbstractArray{<:Any,6})
    tmp = similar(ψ)
    return operate_x!(tmp, dt, ψ)
end

function operate_y(dt::DerivativeTensor, ψ::AbstractArray{<:Any,6})
    tmp = similar(ψ)
    return operate_y!(tmp, dt, ψ)
end

function operate_z(dt::DerivativeTensor, ψ::AbstractArray{<:Any,6})
    tmp = similar(ψ)
    return operate_z!(tmp, dt, ψ)
end


##

mutable struct OperatorSum <: AbstractCompositeOperator
    operators::Vector{AbstractOperator}
    elementgrid::CubicElementGrid
    function OperatorSum(op::AbstractOperator...)
        @assert all(map(x->size(x)==size(s[1]), op))
        new([op...], op[1].elementgrid)
    end
end



function (os::OperatorSum)(ψ::AbstractArray{<:Any,6})
    return sum(op->op(ψ), os.operators)
end
