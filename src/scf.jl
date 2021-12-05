

"""
    helmholtz_equation(Args...; Kwargs...)

Solve Helmholz equation using Greens function.

# Arguments
- `ψ::QuantumState`  :  Initial quess
- `H::Union{HamiltonOperator, HamiltonOperatorMagneticField}`  : Hamilton
    associated with Helmholtz equation that is solved

# Keywords
- `showprogress=false`   :  display progress bar
"""
function helmholtz_equation(ψ::QuantumState, H::HamiltonOperator;
                            showprogress=false)
    normalize!(ψ)
    E = real(bracket(ψ,H,ψ))
    @info "E=$E"
    k  = sqrt( -2(austrip(E)) )
    ct = optimal_coulomb_tranformation(H.elementgrid; k=k);
    ϕ = H.T.m*H.V*1u"ħ_au^-2" * ψ
    ϕ = poisson_equation(ϕ, ct; tmax=ct.tmax, showprogress=showprogress);
    normalize!(ϕ)
    return ϕ
end

function helmholtz_equation!(ψ::QuantumState, H::HamiltonOperator;
                            showprogress=false)
    normalize!(ψ)
    E = real(bracket(ψ,H,ψ))
    @info "E=$E"
    k  = sqrt( -2(austrip(E)) )
    ct = optimal_coulomb_tranformation(H.elementgrid; k=k);
    ϕ = H.T.m*H.V*1u"ħ_au^-2" * ψ
    ψ .= poisson_equation(ϕ, ct; tmax=ct.tmax, showprogress=showprogress);
    normalize!(ψ)
    return ψ
end


function helmholtz_equation(ψ::QuantumState, H::HamiltonOperatorMagneticField;
                            showprogress=false)
    normalize!(ψ)
    E = real(bracket(ψ,H,ψ))
    @info "E=$E"
    k  = sqrt( -2(austrip(E)) )
    ct = optimal_coulomb_tranformation(H.elementgrid; k=k);
    p = momentum_operator(H.T)
    #TODO These could run parallel
    # There is a posible issue with types here
    ϕ = H.T.m * H.V * 1u"ħ_au^-2" * ψ
    ϕ = ϕ + (H.q^2 * u"ħ_au^-2") * (H.A⋅H.A) * ψ
    ϕ = ϕ + (H.q * u"ħ_au^-2") * (H.A⋅p + p⋅H.A) * ψ
    ϕ = poisson_equation(ϕ, ct; tmax=ct.tmax, showprogress=showprogress);
    normalize!(ϕ)
    return ϕ
end


##

"""
    helmholtz_update(Args...; Kwargs...)
    helmholtz_update(sd::SlaterDeterminant, H::AbstractHamiltonOperator; showprogress=false)

Calculate update on Slater Determinant with Helmholtz equation.
"""
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


function helmholtz_update( sd::SlaterDeterminant,
                            H::HamiltonOperatorMagneticField,
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
    p = momentum_operator(H.T)
    ϕ = H.T.m * (H.V + J)* 1u"ħ_au^-2" * sd[1]
    ϕ = ϕ + (H.q^2 * u"ħ_au^-2") * (H.A⋅H.A) * sd[1]
    ϕ = ϕ + (H.q * u"ħ_au^-2") * (H.A⋅p + p⋅H.A) * sd[1]
    tmp = poisson_equation(ϕ, ct; tmax=ct.tmax, showprogress=showprogress);
return normalize!(tmp)
end


function helmholtz_update(sd::SlaterDeterminant, H::HamiltonOperator, J::ScalarOperator, i::Int; nt=96, showprogress=false)
    ct = optimal_coulomb_tranformation(get_elementgrid(sd), nt);
    return helmholtz_update(sd, H, J, i, ct; nt=nt, showprogress=false)
end


function helmholtz_update(sd::SlaterDeterminant, H::AbstractHamiltonOperator, F::AbstractMatrix; showprogress=false, return_fock_matrix=true)
    @argcheck size(F,1) == size(F,2) == length(sd)
    # No preconditioning

    # Update orbitals
    ct = optimal_coulomb_tranformation(sd)
    J = coulomb_operator(sd, ct)
    nsd = pmap(i->helmholtz_update(sd, H, J, i, ct),  axes(sd,1) ) |> SlaterDeterminant
    if return_fock_matrix
        f = fock_matrix(nsd, H)
        ev, ve = eigen( f )
        tmp = SlaterDeterminant( ve'*nsd )
        F = Diagonal(ev)
        return tmp, F
    else
        return nsd
    end
end

function helmholtz_update(sd::SlaterDeterminant, H::AbstractHamiltonOperator; showprogress=false, return_fock_matrix=true)
    #Preconditioning
    f = fock_matrix(sd, H)
    ev, ve = eigen( f )
    nsd = SlaterDeterminant( ve'*sd )
    F = Diagonal(ev)

    return helmholtz_update(nsd, H, F; showprogress=showprogress, return_fock_matrix=return_fock_matrix)
end

##

""" 
    coulomb_operator(Args, Kwargs) -> ScalarOperator

Return Coulomb operator for give `SlaterDeterminant`.
Used density is orbital density (half of electron density).

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
    ρ = density_operator(sd, 1)
    return coulomb_operator(ρ, ct; correction=correction, showprogress=showprogress)
end


function coulomb_operator(density::ScalarOperator, ct::AbstractCoulombTransformation; correction=true, showprogress=false)
    if correction
        ϕ = poisson_equation(density.vals, ct, tmax=ct.tmax, showprogress=showprogress)
    else
        ϕ = poisson_equation(density.vals, ct, showprogress=showprogress)
    end
    # Density is expected to be 0.5 electron density
    # to create literature vesion where F = h + 2J + K and not F = h + J + K
    return ScalarOperator(get_elementgrid(density), ϕ; unit=u"hartree")
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

    #NOTE Fock matrix is calculated as non-symmetric here
    tmp = pmap( axes(sd,1) ) do j
        K = exchange_operator(sd, j, ct)
        f = H + (2J + K)
        map( axes(sd,1) ) do i
            tmp = bracket(sd[i], f, sd[j])
            (austrip∘real)(tmp)
        end
    end
    for (j,col) in zip(axes(out,2), tmp)
        out[:,j] .= col
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


function scf(initial::SlaterDeterminant, H::AbstractHamiltonOperator; max_iter=10, rtol=1E-6)
    @info "Starting scf with $max_iter maximum iterations and $rtol tolerance."
    t = @elapsed begin
        sd, F = helmholtz_update(initial, H)
        E₀ = hf_energy(sd, H, F)
    end
    @info "i = 1,  E = $E₀,  t = $(round(t; digits=1)*u"s")"
    for i in 2:max_iter
        t = @elapsed begin 
            sd, F = helmholtz_update(sd, H, F)
            E = hf_energy(sd, H, F)
        end
        @info "i = $i,  E = $E, t = $(round(t; digits=1)*u"s")"
        if  abs( (E - E₀)/E₀ ) < rtol
            @info "Targeted tolerance archieved. Exiting scf. $(abs(E - E₀)/E₀)" 
            break
        end 
        E₀ = E
    end
    return sd, F
end


"""
    hf_energy(sd::SlaterDeterminant, H::AbstractHamiltonOperator)
    hf_energy(sd::SlaterDeterminant, H::AbstractHamiltonOperator, F::AbstractMatrix)

Calculate Hartree-Fock energy. Fock matrix `F` is expected to be diagonal.
"""
function hf_energy(sd::SlaterDeterminant, H::AbstractHamiltonOperator, F::AbstractMatrix)
    # NOTE expects Diagonal Fock matrix
    return sum( psi->bracket(psi,H,psi), sd ) + sum( diag(F) )*u"hartree" 
end

function hf_energy(sd::SlaterDeterminant, H::AbstractHamiltonOperator)
    F = fock_matrix(sd, H)
    return hf_energy(sd, H, F)
end
