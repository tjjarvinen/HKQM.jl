
"""
    integrate(ϕ, grid::AbstractElementGridSymmetricBox, ψ)
    integrate(grid::AbstractElementGridSymmetricBox, ρ)
    integrate(ϕ::QuantumState, ψ::QuantumState)

Low lever integration routines. (users should not use these, as they can change)
"""
function integrate(ϕ, grid::CubicElementGrid, ψ)
    ω = ω_tensor(grid)
    c = ϕ.*ψ
    return  @tensor ω[α,I]*ω[β,J]*ω[γ,K]*c[α,β,γ,I,J,K]
end

function integrate(ϕ, grid::AbstractElementGridSymmetricBox, ψ)
    ω = getweight(grid)
    c = ϕ.*ψ
    return  @tensor ω[α,I]*ω[β,J]*ω[γ,K]*c[α,β,γ,I,J,K]
end

function integrate(grid::CubicElementGrid, ρ)
    ω = ω_tensor(grid)
    return  @tensor ω[α,I]*ω[β,J]*ω[γ,K]*ρ[α,β,γ,I,J,K]
end

function integrate(grid::AbstractElementGridSymmetricBox, ρ)
    ω = getweight(grid)
    return  @tensor ω[α,I]*ω[β,J]*ω[γ,K]*ρ[α,β,γ,I,J,K]
end

function integrate(ϕ::QuantumState, ψ::QuantumState)
    @assert size(ϕ) == size(ψ)
    integrate(ϕ.psi, ψ.elementgrid, ψ.psi)
end

function integrate(ϕ::QuantumState{Any, Complex}, ψ::QuantumState)
    @assert ϕ.elementgrid == ψ.elementgrid
    integrate(conj(ϕ).psi, ψ.elementgrid, ψ.psi)
end



"""
    bracket(ϕ::QuantumState, ψ::QuantumState)
    bracket(ϕ::QuantumState, op::AbstractOperator, ψ::QuantumState)

Equivalent to Dirac bracket notation <ϕ|ψ> and <ϕ|op|ψ>.
Returns expectation value.
"""
function bracket(ϕ::QuantumState, ψ::QuantumState)
    @assert size(ϕ) == size(ψ)
    return integrate(ϕ.elementgrid, ϕ ⋆ ψ)*unit(ϕ)*unit(ψ)
end

function bracket(ϕ::QuantumState, op::AbstractOperator{1}, ψ::QuantumState)
    @assert size(ϕ) == size(ψ) == size(op)
    return integrate(ϕ.elementgrid, ϕ ⋆ (op*ψ))*unit(ϕ)*unit(ψ)*unit(op)
end

function bracket(ϕ::QuantumState, op::AbstractOperator, ψ::QuantumState)
    @assert size(ϕ) == size(ψ) == size(op)
    return map( O->bracket(ϕ, O, ψ),  op)
end


function bracket(op::AbstractOperator, sd::SlaterDeterminant)
    # One electron operator, so operate on each orbital and multiply by 2
    val = @distributed (+) for i in axes(sd, 1)
        bracket(sd[i], op, sd[i])
    end
    return 2 * val
end

"""
    magnetic_current(ψ::QuantumState, H::HamiltonOperator)
    magnetic_current(ψ::QuantumState, H::HamiltonOperatorMagneticField)
    magnetic_current(sd::SlaterDeterminant, H)

Return magnetic current for systen with given Hamiltonian.

# Returns
`Vector{AbstractArray{Float,6}}`
"""
function magnetic_current(ψ::QuantumState, H::HamiltonOperator)
    p = momentum_operator(H) * (-1u"e_au"/(2H.T.m))
    return _magnetic_current(p, ψ)
end


function magnetic_current(ψ::QuantumState, H::HamiltonOperatorMagneticField)
    p = momentum_operator(H) * (H.q/(2H.T.m))
    return _magnetic_current(p, ψ)
end

function magnetic_current(sd::SlaterDeterminant, H)
    j = sum(sd) do ψ
        magnetic_current(ψ, H)
    end
    return j
end

"""
    para_magnetic_current(ψ::QuantumState)
    para_magnetic_current(sd::SlaterDeterminant)

Calculates para magnetic current.

# Returns
`Vector{AbstractArray{Float,6}}`
"""
function para_magnetic_current(ψ::QuantumState)
    p = momentum_operator(ψ)
    return _magnetic_current(p, ψ)
end

function para_magnetic_current(sd::SlaterDeterminant)
    j = sum(sd) do ψ
        para_magnetic_current(ψ)
    end
    return j
end

function _magnetic_current(p, ψ)
    ϕ = conj(ψ)
    return map( x->real.(ψ⋆(x*ψ) .+ ϕ⋆(x*ϕ)), p)
end

## Coulomb integral / Poisson equation

coulomb_correction(ρ, tmax) = ( 2/sqrt(π)*(π/tmax^2) ) .* ρ



function poisson_equation!(V::AbstractArray{<:Any,6},
                          ρ::AbstractArray{<:Any,6},
                          T::AbstractArray{<:Any,4},
                          wt)
    @assert size(V) == size(ρ)
    @tensoropt V[α,β,γ,I,J,K] = T[α,α',I,I'] * T[β,β',J,J'] * T[γ,γ',K,K'] * ρ[α',β',γ',I',J',K']
    V .*= (2/sqrt(π)*wt)
    return V
end


function poisson_equation(ρ::AbstractArray, transtensor::AbtractTransformationTensor;
                          tmax=nothing, showprogress=false)

    tmp = ρ.*transtensor.wt[1] # Make sure we have correct type
    nt = size(transtensor, 5)
    @debug "nt=$nt"
    ptime = showprogress ? 1 : Inf
    p = Progress(nt, ptime)
    V = sum( axes(transtensor.wt, 1) ) do t
        poisson_equation!(tmp, ρ, transtensor[:,:,:,:,t], transtensor.wt[t])
        next!(p)
        tmp
    end
    if tmax !== nothing
        return V .+ coulomb_correction(ρ, tmax)
    end
    return V
end

function poisson_equation(ψ::QuantumState, transtensor::AbtractTransformationTensor;
                          tmax=nothing, showprogress=false)
    @assert dimension(ψ) == dimension(u"bohr^-2")
    ψ = uconvert(u"bohr^-2", ψ)
    V = poisson_equation(ψ.psi, transtensor, tmax=tmax, showprogress=showprogress)
    return QuantumState(ψ.elementgrid, V)
end
