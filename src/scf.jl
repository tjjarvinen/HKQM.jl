

"""
    helmholtz_equation(Args...; Kwargs...)

Solve Helmholz equation using Greens function.

# Arguments
- `ψ::QuantumState`  :  Initial quess
- `H::Union{HamiltonOperator, HamiltonOperatorMagneticField}`  : Hamilton
    associated with Helmholtz equation that is solved

# Keywords
- `tn=96`      : number of t-integration points
- `tmax=300`   : maximum value for t-integration
- `showprogress=false`   :  display progress bar
"""
function helmholtz_equation(ψ::QuantumState, H::HamiltonOperator;
                            tn=96, tmax=300, showprogress=false)
    normalize!(ψ)
    E = real(bracket(ψ,H,ψ))
    @info "E=$E"
    k  = sqrt( -2(austrip(E)) )
    ct = optimal_coulomb_tranformation(H.elementgrid, tn; k=k);
    ϕ = H.T.m*H.V*1u"ħ_au^-2" * ψ
    ϕ = poisson_equation(ϕ, ct; tmax=tmax, showprogress=showprogress);
    normalize!(ϕ)
    return ϕ
end

function helmholtz_equation!(ψ::QuantumState, H::HamiltonOperator;
                            tn=96, tmax=300, showprogress=false)
    normalize!(ψ)
    E = real(bracket(ψ,H,ψ))
    @info "E=$E"
    k  = sqrt( -2(austrip(E)) )
    ct = optimal_coulomb_tranformation(H.elementgrid, tn; k=k);
    ϕ = H.T.m*H.V*1u"ħ_au^-2" * ψ
    ψ .= poisson_equation(ϕ, ct; tmax=tmax, showprogress=showprogress);
    normalize!(ψ)
    return ψ
end


function helmholtz_equation(ψ::QuantumState, H::HamiltonOperatorMagneticField;
                            tn=96, tmax=300, showprogress=false)
    normalize!(ψ)
    E = real(bracket(ψ,H,ψ))
    @info "E=$E"
    k  = sqrt( -2(austrip(E)) )
    ct = optimal_coulomb_tranformation(H.elementgrid, tn; k=k);
    p = momentum_operator(H.T)
    #TODO These could run parallel
    # There is a posible issue with types here
    ϕ = H.T.m * H.V * 1u"ħ_au^-2" * ψ
    ϕ = ϕ + (H.q^2 * u"ħ_au^-2") * (H.A⋅H.A) * ψ
    ϕ = ϕ + (H.q * u"ħ_au^-2") * (H.A⋅p + p⋅H.A) * ψ
    ϕ = poisson_equation(ϕ, ct; tmax=tmax, showprogress=showprogress);
    normalize!(ϕ)
    return ϕ
end

##

function helmholtz_equation!(sd::SlaterDeterminant, H::HamiltonOperator; tn=96, showprogress=false)
    J = coulomb_operator(sd; showprogress=showprogress)
    E = bracket(sd.orbitals[1], H, sd.orbitals[1]) + bracket(sd.orbitals[1], J, sd.orbitals[1]) |> real
    @info "Orbital energy = $E"
    k  = sqrt( -2(austrip(E)) )
    ct = optimal_coulomb_tranformation(H.elementgrid, tn; k=k);
    ϕ = H.T.m * (H.V + J) * 1u"ħ_au^-2" * sd.orbitals[1]
    sd.orbitals[1] .= poisson_equation(ϕ, ct; tmax=ct.tmax, showprogress=showprogress);
    normalize!(sd.orbitals[1])
    return sd
end

function helmholtz_equation(sd::SlaterDeterminant, H::HamiltonOperatorMagneticField; tn=96, showprogress=false)
    J = coulomb_operator(sd; showprogress=showprogress)
    E = bracket(sd.orbitals[1], H, sd.orbitals[1]) + bracket(sd.orbitals[1], J, sd.orbitals[1]) |> real
    @info "Orbital energy = $E"
    k  = sqrt( -2(austrip(E)) )
    ct = optimal_coulomb_tranformation(H.elementgrid, tn; k=k);
    p = momentum_operator(H.T)
    ϕ = H.T.m * (H.V + J)* 1u"ħ_au^-2" * sd.orbitals[1]
    ϕ = ϕ + (H.q^2 * u"ħ_au^-2") * (H.A⋅H.A) * sd.orbitals[1]
    ϕ = ϕ + (H.q * u"ħ_au^-2") * (H.A⋅p + p⋅H.A) * sd.orbitals[1]
    ϕ = poisson_equation(ϕ, ct; tmax=ct.tmax, showprogress=showprogress);
    normalize!(ϕ)
    return SlaterDeterminant(ϕ)
end

##

function helmholtz_update( sd::SlaterDeterminant,
                           H::HamiltonOperator,
                           J::ScalarOperator,
                           i::Int,
                           ct::AbstractCoulombTransformation;
                           nt=96,
                           showprogress=false
                           )
    Vₑₑ = 2J - exchange_operator(sd, i, ct) 
    E = bracket(sd[i], H, sd[i]) + bracket(sd[1], Vₑₑ, sd[1]) |> real
    k  = sqrt( -2( austrip(E) ) )
    ct = optimal_coulomb_tranformation(H, nt; k=k);
    ϕ = H.T.m * (H.V + Vₑₑ) * 1u"ħ_au^-2" * sd[i]
    tmp = poisson_equation(ϕ, ct; tmax=ct.tmax, showprogress=showprogress);
    return normalize!(tmp)
end

function helmholtz_update(sd::SlaterDeterminant, H::HamiltonOperator, J::ScalarOperator, i::Int; nt=96, showprogress=false)
    ct = optimal_coulomb_tranformation(get_elementgrid(sd), nt);
    return helmholtz_update(sd, H, J, i, ct; nt=nt, showprogress=false)
end


function helmholtz_update(sd::SlaterDeterminant, H::AbstractHamiltonOperator; showprogress=false, nt=96)
    # Preconditioning
    F = fock_matrix(sd, H)
    S = overlap_matrix(sd)
    e, v = eigen(inv(S)*F)
    tmp = SlaterDeterminant(v'*sd)

    # Update orbitals
    ct = optimal_coulomb_tranformation(sd, nt)
    J = coulomb_operator(tmp, ct)
    return pmap(i->helmholtz_update(tmp, H, J, i, ct),  axes(sd,1) ) |> SlaterDeterminant
end


##

""" 
    coulomb_operator(Args, Kwargs) -> ScalarOperator

Return Coulomb operator for give `SlaterDeterminant`

# Args
- `sd::SlaterDeterminant`    :  Orbitals on which the operator is calculated.

# Optional Arguments
- `ct::AbstractCoulombTransformation`   :  transformation tensor, if not given `optimal_coulomb_tranformation` is used

# Kwargs
- `showprogress=false`       : Show progress meter for Poisson equation
- `correction=true`          : Add tail correction to Poisson equation, only if `ct` is given
"""
function coulomb_operator(sd::SlaterDeterminant; showprogress=false)
    ct = optimal_coulomb_tranformation(sd);
    return coulomb_operator(sd, ct; showprogress=showprogress)
end

function coulomb_operator(sd::SlaterDeterminant, ct::AbstractCoulombTransformation; correction=true, showprogress=false)
    #TODO check that this is correct
    ρ = density_operator(sd)
    return coulomb_operator(ρ, ct; correction=correction, showprogress=showprogress)
end


function coulomb_operator(density::ScalarOperator, ct::AbstractCoulombTransformation; correction=true, showprogress=false)
    if correction
        ϕ = poisson_equation(density.vals, ct, tmax=ct.tmax, showprogress=showprogress)
    else
        ϕ = poisson_equation(density.vals, ct, showprogress=showprogress)
    end
    # Multiply by 0.5 to create literature vesion where F = h + 2J + K and not F = h + J + K
    return 0.5 * ScalarOperator(get_elementgrid(density), ϕ; unit=u"hartree")
end


"""
    exchange_operator(Args, Kwargs) -> ScalarOperator

Calculate exchange operator for given `SlaterDeterminant` and orbital index.

# Arguments
- `sd::SlaterDeterminant`  :  Orbitals on which the operator is calculated
- `i::Int`                 :  Orbital index on which exchange operator is calculated

# Optional Arguments
- `ct::AbstractCoulombTransformation`  :  transformation tensor, if not given `optimal_coulomb_tranformation` is used

# Kwargs
- `showprogress=false`     : Show progress meter for Poisson equation solving
"""
function exchange_operator(sd::SlaterDeterminant, i::Int, ct::AbstractCoulombTransformation; showprogress=false)
    @argcheck 0 < i <= length(sd)
    ρ = ketbra( sum( sd.orbitals ), sd.orbitals[i] )
    ϕ = poisson_equation(ρ, ct, tmax=ct.tmax, showprogress=showprogress)
    return ScalarOperator(get_elementgrid(sd), ϕ; unit=u"hartree")
end


function exchange_operator(sd::SlaterDeterminant, i::Int; showprogress=false)
    ct = optimal_coulomb_tranformation(sd)
    return exchange_operator(sd, i, ct; showprogress=showprogress)
end

"""
    fock_matrix(Args, Kwargs) -> Matrix

Return Fock matrix for given orbital space

# Arguments
- `sd::SlaterDeterminant`        :   Orbitals on which Fock matrix is calculated
- `H::AbstractHamiltonOperator`  :   One electron Hamilton

# Optional Arguments
- `ct::AbstractCoulombTransformation`   :  transformation tensor, if not given `optimal_coulomb_tranformation` is used

# Kwargs
- `nt=96`    : number of t-integration points, if `ct` is not given
"""
function fock_matrix(sd::SlaterDeterminant, H::AbstractHamiltonOperator, ct::AbstractCoulombTransformation; showprogress=false)
    p = Progress(length(sd)+1, (showprogress ? 1 : Inf), "Calculating Fock matrix ... ")
    J = coulomb_operator(sd, ct)
    next!(p)
    out = zeros(length(sd), length(sd))
    #TODO This could be parallized
    for j in axes(out,2)
        K = exchange_operator(sd, j, ct)
        f = H + (2J + K)
        for i in axes(out,1)
            tmp = bracket(sd.orbitals[i], f, sd.orbitals[j])
            out[i,j] = real(tmp) |> austrip    # for complex orbitals
        end
        next!(p)
    end
    return out
end


function fock_matrix(sd::SlaterDeterminant, H::AbstractHamiltonOperator)
    ct = optimal_coulomb_tranformation(sd)
    return fock_matrix(sd, H, ct)
end


function overlap_matrix(sd::SlaterDeterminant)
    s = zeros(length(sd), length(sd))
    # TODO consider parallel excecution here
    for i in axes(sd,1)
        for j in i:length(sd)
            s[i,j] = bracket(sd[i], sd[j])
        end
    end
    return Symmetric(s)
end


#= function scf(initial::SlaterDeterminant, H::AbstractHamiltonOperator; nt=96, max_iter=10)
    sd = initial
    ct = optimal_coulomb_tranformation(get_element_grid(H), nt)
    for _ in 1:max_iter
        F = fock_matrix(sd, H)
        e, v = eigen(F)
        tmp = SlaterDeterminant( v'*sd )
        J = coulomb_operator(sd, ct)
        helmholtz_update(tmp, H, J, i)

    end
end =#