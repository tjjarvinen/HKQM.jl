

mutable struct QuantumState{TA,T,N}
    elementgrid::AbstractElementGrid
    psi::TA
    unit::Unitful.FreeUnits
    function QuantumState(aeg::AbstractElementGrid{Taeg, N}, ψ::AbstractArray{T,N}, unit::Unitful.FreeUnits=NoUnits) where {T,N,Taeg}
        @assert size(aeg) == size(ψ)
        new{typeof(ψ), T, N}(aeg, ψ, unit)
    end
end

function Base.show(io::IO, ::MIME"text/plain", ::QuantumState{TA,T,N}) where {TA,T,N}
    print(io, "Quantum state $N dimensions")
end

function Base.show(io::IO, ::QuantumState{TA,T,N}) where {TA,T,N}
    print(io, "Quantum state $N dimensions")
end
Base.size(qs::QuantumState) = size(qs.psi)

Base.getindex(qs::QuantumState, ind...) = qs.psi[ind...]
Base.setindex!(qs::QuantumState, X, ind...) = Base.setindex!(qs.psi, X, ind...)

Base.similar(qs::QuantumState) = QuantumState(qs.elementgrid, similar(qs.psi), qs.unit)

## Unitfull 
Unitful.unit(qs::QuantumState) = qs.unit
Unitful.dimension(qs::QuantumState) = dimension(unit(qs))

function Unitful.uconvert(u::Unitful.Units, qs::QuantumState{TA, T, N}) where {TA,T,N}
    @assert dimension(u) == dimension(qs)
    u == unit(qs) && return qs
    conv = (T ∘ ustrip ∘ uconvert)(u, 1*unit(qs)) * u / unit(qs)
    return conv*qs
end

## utils

get_elementgrid(qs::QuantumState) = qs.elementgrid


## Operations

Base.:(*)(a::Number, qs::QuantumState) = QuantumState(get_elementgrid(qs), ustrip(a).*qs.psi, unit(qs)*unit(a))
Base.:(*)(qs::QuantumState, a::Number) = QuantumState(get_elementgrid(qs), ustrip(a).*qs.psi, unit(qs)*unit(a))
Base.:(/)(qs::QuantumState, a::Number) = QuantumState(get_elementgrid(qs), qs.psi./ustrip(a), unit(qs)/unit(a))

Base.:(*)(a::AbstractVector, qs::QuantumState) = map(x->x*qs, a )
Base.:(-)(qs::QuantumState) = -1*qs


function Base.map(f, qs::QuantumState; output_unit=unit(qs))
    return QuantumState(get_elementgrid(qs), f.(qs.psi), output_unit)
end

function Base.map!(f, qs::QuantumState; output_unit=unit(qs))
    map!(f, qs.psi, qs.psi)
    qs.unit = output_unit
    return qs
end

function Base.map!(f, qs1::QuantumState, qs2::QuantumState; output_unit=unit(qs1))
    map!(f, qs1.psi, qs2.psi)
    qs1.unit = output_unit
    return qs1
end

function Base.map(f, qs1::QuantumState, qs2::QuantumState; output_unit=unit(qs1))
    return QuantumState(get_elementgrid(qs1), f.(qs1.psi, qs2.psi), output_unit)
end

Base.conj(qs::QuantumState) = qs  # dont allocate new here
function Base.conj(qs::QuantumState{TA, T, N}) where {TA, T<:Complex, N}
    return map(Base.conj, qs)
end
Base.conj!(qs::QuantumState) = qs
function Base.conj!(qs::QuantumState{TA, T, N}) where {TA, T<:Complex, N}
    Base.conj!(qs.psi)
    return qs
end



function Base.:(+)(
    qs1::QuantumState{TA1, T1, N}, 
    qs2::QuantumState{TA2, T2, N}
) where{TA1, T1, TA2, T2, N}
    @assert size(qs1) == size(qs2)
    @assert dimension(qs1) == dimension(qs2)
    T = promote_type(T1, T2)
    conv = ustrip(unit(qs1), one(T)*unit(qs2))
    return map(qs1, qs2) do a,b
        muladd(conv, b, a)
    end
end

function Base.:(-)(
    qs1::QuantumState{TA1, T1, N}, 
    qs2::QuantumState{TA2, T2, N}
) where{TA1, T1, TA2, T2, N}
    @assert size(qs1) == size(qs2)
    @assert dimension(qs1) == dimension(qs2)
    T = promote_type(T1, T2)
    conv = -ustrip(unit(qs1), one(T)*unit(qs2))
    return map(qs1, qs2) do a,b
        muladd(conv, b, a)
    end
end



function ketbra(qs1::QuantumState{TA1, T1, N}, qs2::QuantumState{TA2, T2, N}) where{TA1, T1, TA2, T2, N}
    @assert size(qs1) == size(qs2)
    _f(x,y) = conj(x) * y
    _f.(qs1.psi, qs2.psi)
end


function braket(qs1, qs2)
    @assert size(qs1) == size(qs2)
    s = integrate(get_elementgrid(qs1), ketbra(qs1, qs2))
    return s * unit(qs1)*unit(qs2)
end


function normalize!(qs::QuantumState)
    N² = braket(qs, qs) |> ustrip
    if real(N²) > 1e-10
        map!( qs ) do x
            x/√N²
        end
    else
        throw( DivideError() )
    end
    qs.unit = NoUnits
    return qs
end