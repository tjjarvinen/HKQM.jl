using LinearAlgebra


abstract type AbstractQuantumState{T} <: AbstractArray{T,6} end

mutable struct QuantumState{TG,TA,TE} <: AbstractQuantumState{TE}
    elementgrid::TG
    ψ::TA
    function QuantumState(ceg, ψ::AbstractArray{<:Any,6})
        @assert size(ceg) == size(ψ)
        new{typeof(ceg),typeof(ψ),eltype(ψ)}(ceg, ψ)
    end
end


function Base.show(io::IO, ::MIME"text/plain", ::AbstractQuantumState)
    print(io, "Quantum state")
end

Base.size(qs::QuantumState) = size(qs.ψ)


Base.getindex(qs::QuantumState, ind...) = qs.ψ[ind...]
Base.setindex!(qs::QuantumState, X, ind...) = Base.setindex!(qs.ψ, X, ind...)

Base.similar(qs::QuantumState) = QuantumState(qs.elementgrid, qs.ψ)


function normalize!(qs::QuantumState)
    N² = integrate(qs, qs)
    qs .*= 1/√N²
    return qs
end


function Base.:+(qs1::QuantumState{T,<:Any,<:Any}, qs2::QuantumState{T,<:Any,<:Any}) where T
    return QuantumState(qs1.elementgrid, qs1.ψ.+qs2.ψ)
end

function Base.:-(qs1::QuantumState{T,<:Any,<:Any}, qs2::QuantumState{T,<:Any,<:Any}) where T
    return QuantumState(qs1.elementgrid, qs1.ψ.-qs2.ψ)
end
