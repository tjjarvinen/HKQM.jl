

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
- `tn=96`                    : Number of t-integration points, if `ct` is not given
"""
function coulomb_operator(sd::SlaterDeterminant; tn=96, showprogress=false)
    ct = optimal_coulomb_tranformation(get_elementgrid(sd), tn);
    return coulomb_operator(sd, ct; showprogress=showprogress)
end

function coulomb_operator(sd::SlaterDeterminant, ct::AbstractCoulombTransformation; correction=true, showprogress=false)
    #TODO check that this is correct
    ρ = density_operator(sd, 1)
    if correction
        ϕ = poisson_equation(ρ.vals, ct, tmax=ct.tmax, showprogress=showprogress)
    else
        ϕ = poisson_equation(ρ.vals, ct, showprogress=showprogress)
    end
    return ScalarOperator(get_elementgrid(sd), ϕ; unit=u"hartree")
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
- `nt=96`                  : number of t-integration points, when `ct` is not given
"""
function exchange_operator(sd::SlaterDeterminant, i::Int, ct::AbstractCoulombTransformation; showprogress=false)
    @argcheck 0 < i <= length(sd)
    ρ = ketbra( sum( sd.orbitals ), sd.orbitals[i] )
    ϕ = poisson_equation(ρ, ct, tmax=ct.tmax, showprogress=showprogress)
    return ScalarOperator(get_elementgrid(sd), ϕ; unit=u"hartree")
end


function exchange_operator(sd::SlaterDeterminant, i::Int; nt=96, showprogress=false)
    ct = optimal_coulomb_tranformation(get_elementgrid(sd), nt)
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
function fock_matrix(sd::SlaterDeterminant, H::AbstractHamiltonOperator, ct::AbstractCoulombTransformation)
    p = Progress(length(sd)+1, 1, "Calculating Fock matrix ... ")
    J = coulomb_operator(sd, ct)
    next!(p)
    out = zeros(length(sd), length(sd))
    for j in axes(out,2)
        K = exchange_operator(sd, j, ct)
        f = H + (2J + K)
        for i in axes(out,1)
            tmp = bracket(sd.orbitals[i], f, sd.orbitals[j])
            out[i,j] = real(tmp) |> austrip
        end
        next!(p)
    end
    return out
end


function fock_matrix(sd::SlaterDeterminant, H::AbstractHamiltonOperator; nt=96)
    ct = optimal_coulomb_tranformation(get_elementgrid(sd), nt)
    return fock_matrix(sd, H, ct)
end