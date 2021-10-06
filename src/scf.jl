

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

function coulomb_operator(sd::SlaterDeterminant; tn=96, showprogress=false)
    ct = optimal_coulomb_tranformation(get_elementgrid(sd), tn);
    return coulomb_operator(sd, ct; showprogress=showprogress)
end

function coulomb_operator(sd::SlaterDeterminant, ct::AbstractCoulombTransformation; correction=true, showprogress=false)
    occupations = fill(2, length(sd))
    occupations[1] = 1
    ρ = density_operator(sd, occupations)
    if correction
        ϕ = poisson_equation(ρ.vals, ct, tmax=ct.tmax, showprogress=showprogress)
    else
        ϕ = poisson_equation(ρ.vals, ct, showprogress=showprogress)
    end
    return ScalarOperator(get_elementgrid(sd), ϕ; unit=u"hartree")
end