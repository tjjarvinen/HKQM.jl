

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
    ψ = sd[i]
    Kψ = exchange_operator(sd, i, ct)
    E = bracket(ψ, H, ψ) + 2*bracket(ψ, J, ψ) - bracket(ψ, Kψ) |> real
    k  = sqrt( -2( austrip(E) ) )
    ct = optimal_coulomb_tranformation(H, nt; k=k);
    ϕ = (H.V + 2J) * ψ - Kψ  
    ϕ *= H.T.m * 1u"ħ_au^-2"
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
    ψ = sd[i]
    Kψ = exchange_operator(sd, i, ct)  #TODO This is wrong ψᵢ is now put in two times
    E = bracket(ψ, H, ψ) + 2*bracket(ψ, J, ψ) - bracket(ψ, Kψ) |> real
    k  = sqrt( -2( austrip(E) ) )
    ct = optimal_coulomb_tranformation(H, nt; k=k);
    p = momentum_operator(H.T)
    ϕ = (H.V + 2J) * ψ - Kψ  
    ϕ *= H.T.m * 1u"ħ_au^-2"
    ϕ += (H.q^2 * u"ħ_au^-2") * (H.A⋅H.A) * ψ
    ϕ += (H.q * u"ħ_au^-2") * (H.A⋅p + p⋅H.A) * ψ
    tmp = poisson_equation(ϕ, ct; tmax=ct.tmax, showprogress=showprogress);
return normalize!(tmp)
end


function helmholtz_update(sd::SlaterDeterminant, H::AbstractHamiltonOperator, J::ScalarOperator, i::Int; nt=96, showprogress=false)
    ct = optimal_coulomb_tranformation(get_elementgrid(sd), nt);
    return helmholtz_update(sd, H, J, i, ct; nt=nt, showprogress=false)
end



function helmholtz_update(sd::SlaterDeterminant, H::AbstractHamiltonOperator; showprogress=false)
    # To do calculation the whole Slater determinant needs to be moved to everywhere.
    # Because exchange integral needs it.
    # Thus is better to calculate Coulomb operator differently on each node.
    # and save a little bit on transfer time.
    @debug "Number of progs in helmholtz_update $(nprocs())"
    if nprocs() == 1
        @debug "Doing non parallel helmholtz_update"
        J = coulomb_operator(sd)
        tmp = map( axes(sd,1) ) do i
            helmholtz_update(sd, H, J, i)
        end
    else
        @debug "Doing parallel helmholtz_update"
        tmp = pmap( axes(sd,1) ) do i
            J = coulomb_operator(sd)
            helmholtz_update(sd, H, J, i)
        end
    end
    return SlaterDeterminant(tmp)
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
    # Given total charge density
    # Calculate electric potential and multiply it with charge
    # To get potential energy
    if correction
        ϕ = poisson_equation(density.vals, ct, tmax=ct.tmax, showprogress=showprogress)
    else
        ϕ = poisson_equation(density.vals, ct, showprogress=showprogress)
    end
    # Density is expected to be 0.5 * electron density
    # to create literature vesion where F = h + 2J + K and not F = h + J + K
    # Electron charge is expected to be -1
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
"""
function exchange_operator(sd::SlaterDeterminant, i::Int, ct::AbstractCoulombTransformation; showprogress=false)
    # TODO this should be ok
    @argcheck 0 < i <= length(sd)
    # ΣⱼKⱼψᵢ
    ψ = sum(sd) do ϕ
        ρ = ketbra( ϕ, sd[i] )
        tmp = poisson_equation(ρ, ct, tmax=ct.tmax)
        ScalarOperator(get_elementgrid(sd), tmp; unit=u"hartree") * ϕ
    end
    return ψ
end


function exchange_operator(sd::SlaterDeterminant, i::Int; showprogress=false)
    ct = optimal_coulomb_tranformation(sd)
    return exchange_operator(sd, i, ct; showprogress=showprogress)
end


"""
    fock_matrix(Args, Kwargs) -> Matrix

Return Fock matrix for given orbital space. The matrix is not necessary Hermiations, to allow checks for calculations.

# Arguments
- `sd::SlaterDeterminant`        :   Orbitals on which Fock matrix is calculated
- `H::AbstractHamiltonOperator`  :   One electron Hamilton

# Optional Arguments
- `ct::AbstractCoulombTransformation`   :  transformation tensor, if not given `optimal_coulomb_tranformation` is used

# Kwargs
- `nt=96`    : number of t-integration points, if `ct` is not given
"""
function fock_matrix(sd::SlaterDeterminant, H::AbstractHamiltonOperator, ct::AbstractCoulombTransformation; showprogress=false)
    J = coulomb_operator(sd, ct)
    F = pmap( axes(sd,1) ) do j
        Kψ = exchange_operator(sd, j)  # Might be ok the send in ct, but might be a waste of bandwith
        map( axes(sd,1) ) do i  #TODO Fock matrix is symmetric, thus here is a little extra
            # F = h + 2J - K
            bracket(sd[i], H, sd[j]) + 2*bracket(sd[i], J, sd[j]) - bracket(sd[i], Kψ) 
        end
    end
    return hcat(F...)
end


function fock_matrix(sd::SlaterDeterminant, H::AbstractHamiltonOperator)
    ct = optimal_coulomb_tranformation(sd)
    return fock_matrix(sd, H, ct)
end


function overlap_matrix(sd::SlaterDeterminant)
    #TODO Make this calculation to use symmetricity of the result
    # Also some kind of parallelization might work here
    S = map( Iterators.product(axes(sd,1), axes(sd,1)) ) do (i,j)
            bracket(sd[i], sd[j])
    end
    return S
end


function scf!(sd::SlaterDeterminant, H::AbstractHamiltonOperator; max_iter::Int=10, tol=1E-6)
    @argcheck tol > 0
    @argcheck max_iter > 0
    function _check_overlap(sd0, sd1)
        tmp = (abs ∘ bracket).(sd0, sd1) |> minimum
        return round(1 - tmp; sigdigits=2)
    end
    @info "Starting scf with $max_iter maximum iterations and $tol tolerance."
    t = @elapsed begin
        sdn = helmholtz_update(sd, H)
        dS = _check_overlap(sd, sdn)
        sd = sdn
    end
    @info "i = 1,  ΔS = $dS,  t = $(round(t; digits=1)*u"s")"
    for i in 2:max_iter
        t = @elapsed begin 
            sdn = helmholtz_update(sd, H)
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


"""
    hf_energy(sd::SlaterDeterminant, H::AbstractHamiltonOperator)
    hf_energy(sd::SlaterDeterminant, H::AbstractHamiltonOperator, F::AbstractMatrix)

Calculate Hartree-Fock energy. Fock matrix `F` is expected to be diagonal.
"""
function hf_energy(sd::SlaterDeterminant, H::AbstractHamiltonOperator, F::AbstractMatrix)
    # NOTE expects Diagonal Fock matrix
    return sum( psi->bracket(psi,H,psi), sd ) + sum( diag(F) )
end

function hf_energy(sd::SlaterDeterminant, H::AbstractHamiltonOperator)
    F = fock_matrix(sd, H)
    return hf_energy(sd, H, F)
end
