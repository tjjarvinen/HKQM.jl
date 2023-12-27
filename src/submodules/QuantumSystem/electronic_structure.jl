
density_operator(so::ScalarOperator) = ScalarOperator(get_elementgrid(so), so.vals)
density_operator(qs::QuantumState) = ScalarOperator(get_elementgrid(qs), ketbra(qs, qs))
function charge_density(qs; charge=-1u"e_au")
    @argcheck dimension(charge) == dimension(u"C")
    return density_operator(qs) * charge
end


function electric_potential(cdensity::ScalarOperator{TA, T, 3}; correction=true) where {TA, T}
    @argcheck dimension(cdensity) == dimension(u"C")
    ega = get_elementgrid(cdensity)
    @argcheck dimension(ega) == dimension(u"m")
    ρ = auconvert(cdensity)
    tx = default_transformation_tensor(ega, 1)
    ty = default_transformation_tensor(ega, 2)
    tz = default_transformation_tensor(ega, 3)
    tmp = apply_transformation(ρ.vals, tx, ty, tz; correction=correction)
    return ScalarOperator(ega, tmp; unit=u"hartree/e_au"*unit(ega)^2/u"bohr"^2) |> auconvert
end


##


abstract type AbstractSlaterDeterminant <: AbstractVector{QuantumState} end


struct SlaterDeterminant <: AbstractSlaterDeterminant
    orbitals::Vector{QuantumState}
    function SlaterDeterminant(orbitals::AbstractVector; orthogonalize=true)
        function _gram_schmit(orbitals::AbstractVector)
            out = []
            push!(out, orbitals[begin])
            for i in 2:length(orbitals)
                tmp = sum( x -> braket(out[x], orbitals[i])*out[x], 1:i-1 )
                push!(out, orbitals[i] - tmp)
            end
            return normalize!.(out)
        end
        # We are changing orbitals so we need to take deepcopy
        tmp = deepcopy.(orbitals)
        if orthogonalize    
            new( _gram_schmit(tmp) )
        else
            new( normalize.(tmp) )
        end
    end
end

function SlaterDeterminant(orbitals::QuantumState...)
    return SlaterDeterminant(collect(orbitals))
end 


function Base.show(io::IO, ::MIME"text/plain", s::SlaterDeterminant)
    print(io, "SlaterDetermiant $(length(s.orbitals)) orbitals")
end

function Base.show(io::IO, s::SlaterDeterminant)
    print(io, "SlaterDetermiant $(length(s.orbitals)) orbitals")
end

Base.size(sd::SlaterDeterminant) = size(sd.orbitals)

get_elementgrid(sd::SlaterDeterminant) = get_elementgrid(sd.orbitals[1])

Base.getindex(sd::SlaterDeterminant, i) = sd.orbitals[i] 