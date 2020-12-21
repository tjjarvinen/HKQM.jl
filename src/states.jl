using LinearAlgebra


abstract type AbstractQuantumState{T} <: AbstractArray{T,6} end

mutable struct QuantumState{TG,TA,TE} <: AbstractQuantumState{TE}
    elementgrid::TG
    ψ::TA
    unit::Unitful.FreeUnits
    function QuantumState(ceg, ψ::AbstractArray{<:Any,6}, unit::Unitful.FreeUnits=NoUnits)
        @assert size(ceg) == size(ψ)
        new{typeof(ceg),typeof(ψ),eltype(ψ)}(ceg, ψ, unit)
    end
end


function Base.show(io::IO, ::MIME"text/plain", ::AbstractQuantumState)
    print(io, "Quantum state")
end

Base.size(qs::QuantumState) = size(qs.ψ)


Base.getindex(qs::QuantumState, ind...) = qs.ψ[ind...]
Base.setindex!(qs::QuantumState, X, ind...) = Base.setindex!(qs.ψ, X, ind...)

Base.similar(qs::QuantumState) = QuantumState(qs.elementgrid, qs.ψ)
Unitful.unit(qs::QuantumState) = qs.unit
Unitful.dimension(qs::QuantumState) = dimension(unit(qs))

function Unitful.uconvert(u::Unitful.Units, qs::QuantumState)
    @assert dimension(u) == dimension(qs)
    u == unit(qs) && return qs
    conv = ustrip(uconvert(u, 1*unit(qs))) * u / unit(qs)
    return conv*qs
end

function normalize!(qs::QuantumState)
    N² = bracket(qs, qs)
    qs.ψ .*= 1/√N²
    return qs
end


function Base.:(+)(qs1::QuantumState{T,<:Any,<:Any}, qs2::QuantumState{T,<:Any,<:Any}) where T
    @assert dimension(qs1) == dimension(qs2)
    @assert size(qs1) == size(qs2)
    if unit(qs1) == unit(qs2)
        return QuantumState(qs1.elementgrid, qs1.ψ.+qs2.ψ, unit(qs1))
    else
        return qs1 + uconvert(unit(qs1), qs2)
    end
end

function Base.:(-)(qs1::QuantumState{T,<:Any,<:Any}, qs2::QuantumState{T,<:Any,<:Any}) where T
    @assert dimension(qs1) == dimension(qs2)
    @assert size(qs1) == size(qs2)
    return QuantumState(qs1.elementgrid, qs1.ψ.-qs2.ψ, unit(qs1))
end

"""
    ⋆(qs1::QuantumState{T,<:Any,<:Any}, qs2::QuantumState{T,<:Any,<:Any})

Return probability density ψ†ψ

## Example
julia> ψ⋆ψ
"""
function ⋆(qs1::QuantumState{T,<:Any,<:Any}, qs2::QuantumState{T,<:Any,<:Any}) where T
    @assert size(qs1) == size(qs2)
    conj(qs1).ψ .* qs2.ψ
end


Base.:(*)(a::Number, qs::QuantumState) = QuantumState(qs.elementgrid, ustrip(a).*qs.ψ, unit(qs)*unit(a))
Base.:(*)(qs::QuantumState, a::Number) = QuantumState(qs.elementgrid, ustrip(a).*qs.ψ, unit(qs)*unit(a))
Base.:(/)(qs::QuantumState, a::Number) = QuantumState(qs.elementgrid, qs.ψ./ustrip(a), unit(qs)/unit(a))

Base.conj(qs::QuantumState) = qs
Base.conj(qs::QuantumState{Any, Any, Complex}) = QuantumState(qs.elementgrid, conj.(qs.ψ), unit(qs))
Base.conj!(qs::QuantumState) = qs
function Base.conj!(qs::QuantumState{Any, Any, Complex})
    conj!.(qs.ψ)
    return qs
end
