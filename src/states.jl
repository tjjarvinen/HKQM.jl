



"""
    QuantumState{TG,TA,TE}

Type to hold and manipulate quantum states

# Fields
- `elementgrid::TG`  : grid where quantum state is "=basis"
- `psi::TA`            : array holding "values"
- `unit::Unitful.FreeUnits`  : unit for quantum state

# Creation
    QuantumState(ceg, ψ::AbstractArray{<:Any,6}, unit::Unitful.FreeUnits=NoUnits)

# Example
```jldoctest
julia> ceg = CubicElementGrid(5, 2, 32)
Cubic elements grid with 2^3 elements with 32^3 Gauss points

julia> ψ = QuantumState(ceg, ones(size(ceg)))
Quantum state

julia> normalize!(ψ)
Quantum state

julia> bracket(ψ, ψ) ≈ 1
true

julia> ϕ = ψ + 2ψ
Quantum state

julia> bracket(ψ, ϕ) ≈ 3
true

julia> unit(ψ)


julia> unit(1u"bohr"*ψ)
a₀
```
"""
mutable struct QuantumState{TG,TA,TE} <: AbstractQuantumState{TE}
    elementgrid::TG
    psi::TA
    unit::Unitful.FreeUnits
    function QuantumState(ceg, ψ::AbstractArray{<:Any,6}, unit::Unitful.FreeUnits=NoUnits)
        @assert size(ceg) == size(ψ)
        new{typeof(ceg),typeof(ψ),eltype(ψ)}(ceg, ψ, unit)
    end
end

QuantumState(op::AbstractScalarOperator) = QuantumState(get_elementgrid(op), get_values(op), unit(op))


function Base.show(io::IO, ::MIME"text/plain", ::AbstractQuantumState)
    print(io, "Quantum state")
end

Base.size(qs::QuantumState) = size(qs.psi)


Base.getindex(qs::QuantumState, ind...) = qs.psi[ind...]
Base.setindex!(qs::QuantumState, X, ind...) = Base.setindex!(qs.psi, X, ind...)

Base.similar(qs::QuantumState) = QuantumState(qs.elementgrid, qs.psi)
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
    if real(N²) > 1e-10
        qs.psi .*= 1/√N²
    end
    return qs
end


function Base.:(+)(qs1::QuantumState{T,<:Any,<:Any}, qs2::QuantumState{T,<:Any,<:Any}) where T
    @assert dimension(qs1) == dimension(qs2)
    @assert size(qs1) == size(qs2)
    if unit(qs1) == unit(qs2)
        return QuantumState(qs1.elementgrid, qs1.psi.+qs2.psi, unit(qs1))
    else
        return qs1 + uconvert(unit(qs1), qs2)
    end
end

function Base.:(-)(qs1::QuantumState{T,<:Any,<:Any}, qs2::QuantumState{T,<:Any,<:Any}) where T
    @assert dimension(qs1) == dimension(qs2)
    @assert size(qs1) == size(qs2)
    return QuantumState(qs1.elementgrid, qs1.psi.-qs2.psi, unit(qs1))
end

"""
    ⋆(qs1::QuantumState{T,<:Any,<:Any}, qs2::QuantumState{T,<:Any,<:Any})
    ketbra(qs1::QuantumState{T,<:Any,<:Any}, qs2::QuantumState{T,<:Any,<:Any})

Return probability density ψ†ψ = |ψ><ψ|

## Example
```julia
julia> ψ⋆ψ
```
"""
function ketbra(qs1::QuantumState{T,<:Any,<:Any}, qs2::QuantumState{T,<:Any,<:Any}) where T
    @assert size(qs1) == size(qs2)
    conj(qs1).psi .* qs2.psi
end

const ⋆ = ketbra

Base.:(*)(a::Number, qs::QuantumState) = QuantumState(qs.elementgrid, ustrip(a).*qs.psi, unit(qs)*unit(a))
Base.:(*)(qs::QuantumState, a::Number) = QuantumState(qs.elementgrid, ustrip(a).*qs.psi, unit(qs)*unit(a))
Base.:(/)(qs::QuantumState, a::Number) = QuantumState(qs.elementgrid, qs.psi./ustrip(a), unit(qs)/unit(a))

Base.conj(qs::QuantumState) = qs
Base.conj(qs::QuantumState{Any, Any, Complex}) = QuantumState(qs.elementgrid, conj.(qs.psi), unit(qs))
Base.conj!(qs::QuantumState) = qs
function Base.conj!(qs::QuantumState{Any, Any, Complex})
    conj!.(qs.psi)
    return qs
end


get_elementgrid(qs::QuantumState) = qs.elementgrid



##

abstract type AbstractSlaterDeterminant end


struct SlaterDeterminant <: AbstractSlaterDeterminant
    orbitals::Vector{QuantumState}
    function SlaterDeterminant(orbitals::QuantumState...)
        new(collect(orbitals))
    end
end

function Base.show(io::IO, ::MIME"text/plain", s::SlaterDeterminant)
    print(io, "SlaterDetermiant $(length(s.orbitals)) orbitals")
end

Base.length(sd::SlaterDeterminant) = length(sd.orbitals)

get_elementgrid(sd::SlaterDeterminant) = get_elementgrid(sd.orbitals[1])