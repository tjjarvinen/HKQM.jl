
density_operator(so::ScalarOperator) = ScalarOperator(get_elementgrid(so), so.vals)
density_operator(qs::QuantumState) = ScalarOperator(get_elementgrid(qs), ketbra(qs, qs))

function density_operator(qs1::QuantumState, qs2::QuantumState)
    return ScalarOperator(get_elementgrid(qs1), ketbra(qs1, qs2))
end

function charge_density(qs; charge=-1u"e_au")
    @argcheck dimension(charge) == dimension(u"C")
    return density_operator(qs) * charge
end


function electric_potential(cdensity::ScalarOperator{TA, T, 3}; correction=true) where {TA, T}
    @argcheck dimension(cdensity) == dimension(u"C") || dimension(cdensity) == dimension(1.0)
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

density_operator(sd::SlaterDeterminant) = sum( density_operator, sd.orbitals )

function Base.show(io::IO, ::MIME"text/plain", s::SlaterDeterminant)
    print(io, "SlaterDetermiant $(length(s.orbitals)) orbitals")
end

function Base.show(io::IO, s::SlaterDeterminant)
    print(io, "SlaterDetermiant $(length(s.orbitals)) orbitals")
end

Base.size(sd::SlaterDeterminant) = size(sd.orbitals)

get_elementgrid(sd::SlaterDeterminant) = get_elementgrid(sd.orbitals[1])

Base.getindex(sd::SlaterDeterminant, i) = sd.orbitals[i] 


##

function coulomb_operator(sd::SlaterDeterminant)
    ρ = density_operator(sd)
    ϕ = electric_potential(ρ)
    return ScalarOperator(ϕ; unit=u"hartree")
end


function exchange_operator(sd::SlaterDeterminant, i::Integer)
    @argcheck 0 < i <= length(sd)
    ψ = sum(sd) do ϕ
        ρ = density_operator(ϕ, sd[i])
        tmp = electric_potential(ρ)
        ScalarOperator(tmp; unit=u"hartree") * ϕ
    end
    return ψ
end


function fock_matrix(sd::SlaterDeterminant, H::AbstractHamiltonOperator)
    J = coulomb_operator(sd)
    F = pmap( axes(sd,1) ) do j
        Kψⱼ = exchange_operator(sd, j)
        tmp = map( axes(sd,1) ) do i  #TODO Fock matrix is symmetric, thus here is a little extra
            # F = h + 2J - K
            braket(sd[i], H, sd[j]) + 2*braket(sd[i], J, sd[j]) - braket(sd[i], Kψⱼ) 
        end
        tmp
    end
    return reduce(hcat, F)
end


function overlap_matrix(sd::SlaterDeterminant)
    #TODO Make this calculation to use symmetricity of the result
    # Also some kind of parallelization might work here
    S = map( Iterators.product(axes(sd,1), axes(sd,1)) ) do (i,j)
            braket(sd[i], sd[j])
    end
    return S
end



function helmholtz_equation(sd::SlaterDeterminant, H::HamiltonOperator, J::ScalarOperator, i::Integer)
    ψ = sd[i]
    Kψ = exchange_operator(sd, i)
    E = braket(ψ, H, ψ) + 2*braket(ψ, J, ψ) - braket(ψ, Kψ) |> real
    if E >= 0u"hartree"
        DomainError("Orbital $i has positive energy") |> throw
    end
    k = sqrt(-2H.T.m * E)/u"ħ_au"
    ϕ = (H.V + 2J) * ψ - Kψ  
    ϕ *= H.T.m * 1u"ħ_au^-2"
    tmp = helmholtz_equation(ϕ, k)
    return normalize!(tmp)
end


function helmholtz_equation(sd::SlaterDeterminant, H::AbstractHamiltonOperator)
    J = coulomb_operator(sd)
    tmp = pmap( axes(sd,1) ) do i
        helmholtz_equation(sd, H, J, i)
    end
    return SlaterDeterminant(tmp)
end


function scf!(sd::SlaterDeterminant, H::AbstractHamiltonOperator; max_iter::Int=10, tol=1E-6)
    @argcheck tol > 0
    @argcheck max_iter > 0
    function _check_overlap(sd0, sd1)
        tmp = (abs ∘ braket).(sd0, sd1) |> minimum
        return round(1 - tmp; sigdigits=2)
    end
    @info "Starting scf with $max_iter maximum iterations and $tol tolerance."
    t = @elapsed begin
        sdn = helmholtz_equation(sd, H)
        dS = _check_overlap(sd, sdn)
        sd = sdn
    end
    @info "i = 1,  ΔS = $dS,  t = $(round(t; digits=1)*u"s")"
    for i in 2:max_iter
        t = @elapsed begin 
            sdn = helmholtz_equation(sd, H)
            dS = _check_overlap(sd, sdn)
            sd = sdn
        end
        @info "i = $i,  ΔS = $dS, t = $(round(t; digits=1)*u"s")"
        if  dS < tol
            @info "Targeted tolerance archieved. Exiting scf. $dS < $tol" 
            break
        end 
    end
    return sd
end


function scf(sd::SlaterDeterminant, H::AbstractHamiltonOperator; max_iter=10, tol=1E-6)
    tmp = deepcopy(sd)
    return scf!(tmp::SlaterDeterminant, H::AbstractHamiltonOperator; max_iter=max_iter, tol=tol)    
end



function hf_energy(sd::SlaterDeterminant, H::AbstractHamiltonOperator, F::AbstractMatrix)
    # NOTE expects Diagonal Fock matrix
    return sum( psi->braket(psi,H,psi), sd ) + sum( diag(F) )
end

function hf_energy(sd::SlaterDeterminant, H::AbstractHamiltonOperator)
    F = fock_matrix(sd, H)
    return hf_energy(sd, H, F)
end