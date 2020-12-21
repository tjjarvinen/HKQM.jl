using Distributed


function integrate(ϕ, grid::CubicElementGrid, ψ)
    ω = ω_tensor(grid)
    c = ϕ.*ψ
    return  @tensor ω[α,I]*ω[β,J]*ω[γ,K]*c[α,β,γ,I,J,K]
end

function integrate(grid::CubicElementGrid, ρ)
    ω = ω_tensor(grid)
    return  @tensor ω[α,I]*ω[β,J]*ω[γ,K]*ρ[α,β,γ,I,J,K]
end

function integrate(ϕ::QuantumState, ψ::QuantumState)
    @assert ϕ.elementgrid == ψ.elementgrid
    integrate(ϕ.ψ, ψ.elementgrid, ψ.ψ)
end

function integrate(ϕ::QuantumState{Any, Any, Complex}, ψ::QuantumState)
    @assert ϕ.elementgrid == ψ.elementgrid
    integrate(conj.(ϕ.ψ), ψ.elementgrid, ψ.ψ)
end

function bracket(ϕ::QuantumState, ψ::QuantumState)
    @assert ϕ.elementgrid == ψ.elementgrid
    return integrate(ϕ.elementgrid, ϕ ⋆ ψ)*unit(ϕ)*unit(ψ)
end

function bracket(ϕ::QuantumState, op::AbstractOperator{1}, ψ::QuantumState)
    @assert ϕ.elementgrid == ψ.elementgrid == op.elementgrid
    return integrate(ϕ.elementgrid, ϕ ⋆ (op*ψ))*unit(ϕ)*unit(ψ)*unit(op)
end

function bracket(ϕ::QuantumState, op::AbstractOperator, ψ::QuantumState)
    @assert ϕ.elementgrid == ψ.elementgrid == op.elementgrid
    return pmap( O->bracket(ϕ, O, ψ),  op)
end


function magnetic_current(ψ::QuantumState, H::HamiltonOperator)
    p = momentum_operator(H) * (-1u"e_au"/(2H.T.m))
    ϕ = conj(ψ)
    jx = ψ⋆(p[1]*ψ) .+ ϕ⋆(p[1]*ϕ)
    jy = ψ⋆(p[2]*ψ) .+ ϕ⋆(p[2]*ϕ)
    jz = ψ⋆(p[3]*ψ) .+ ϕ⋆(p[3]*ϕ)
    return [real.(jx), real.(jy), real.(jz)]
end

function magnetic_current(ψ::QuantumState, H::HamiltonOperatorMagneticField)
    p = momentum_operator(H) * (H.q/(2H.T.m))
    ϕ = conj(ψ)
    jx = ψ⋆(p[1]*ψ) .+ ϕ⋆(p[1]*ϕ)
    jy = ψ⋆(p[2]*ψ) .+ ϕ⋆(p[2]*ϕ)
    jz = ψ⋆(p[3]*ψ) .+ ϕ⋆(p[3]*ϕ)
    return [real.(jx), real.(jy), real.(jz)]
end

## Coulomb integral / Poisson equation

function coulomb_tensor(ρ::AbstractArray, transtensor::AbtractTransformationTensor;
                        tmax=nothing, showprogress=false)
    V = similar(ρ)
    V .= 0
    v = similar(ρ)
    ptime = showprogress ? 1 : Inf
    @showprogress ptime "Calculating v-tensor..." for p in Iterators.reverse(eachindex(transtensor.wt))
        T = transtensor[:,:,:,:,p]
        @tensoropt v[α,β,γ,I,J,K] = T[α,α',I,I']*T[β,β',J,J']*T[γ,γ',K,K']*ρ[α',β',γ',I',J',K']
        V = V .+ transtensor.wt[p].*v
    end
    if tmax != nothing
        return muladd.(2/sqrt(π), V, coulomb_correction(ρ, tmax))
    end
    return 2/sqrt(π).*V
end

function coulomb_tensor(ρ::AbstractArray, transtensor::AbstractArray, wt::AbstractVector;
                        tmax=nothing, showprogress=false)
    @assert size(transtensor)[end] == length(wt)
    V = similar(ρ)
    V .= 0
    v = similar(ρ)
    ptime = showprogress ? 1 : Inf
    @showprogress ptime "Calculating v-tensor..." for p in Iterators.reverse(eachindex(wt))
        T = @view transtensor[:,:,:,:,p]
        @tensoropt v[α,β,γ,I,J,K] = T[α,α',I,I']*T[β,β',J,J']*T[γ,γ',K,K']*ρ[α',β',γ',I',J',K']
        V = V .+ wt[p].*v
    end
    if tmax != nothing
        return muladd.(2/sqrt(π), V, coulomb_correction(ρ, tmax))
    end
    return  2/sqrt(π).*V
end


coulomb_correction(ρ, tmax) = 2/sqrt(π)*(π/tmax^2).*ρ

function poisson_equation(ρ::AbstractArray,
                          transtensor::AbtractTransformationTensor,
                          I, J, K;
                          tmax=nothing)
    V = similar(ρ, size(ρ)[1:3]...)
    V .= 0
    v = similar(V)
    @showprogress for p in Iterators.reverse(eachindex(transtensor.wt))
        Tx = transtensor[:,:,:,I,p]
        Ty = transtensor[:,:,:,J,p]
        Tz = transtensor[:,:,:,K,p]
        @tensoropt v[α,β,γ] = Tx[α,α',I'] * Ty[β,β',J'] * Tz[γ,γ',K'] * ρ[α',β',γ',I',J',K']
        #@tensoropt v[α',β',γ',I',J',K'] = Tx[α,α',I'] * Ty[β,β',J'] * Tz[γ,γ',K'] * vρ[α,β,γ]
        V = V .+ transtensor.wt[p].*v
    end
    if tmax != nothing
        return muladd.(2/sqrt(π), V, coulomb_correction(ρ[:,:,:,I,J,K], tmax))
    end
    return  2/sqrt(π).*V
end


function poisson_equation(ρ::AbstractArray{<:Any,6},
                          transtensor::AbtractTransformationTensor,
                          t)
    V = similar(ρ)
    T = similar(ρ, size(transtensor)[1:end-1])
    T .= transtensor[:,:,:,:,t]
    @tensoropt V[α,β,γ,I,J,K] = T[α,α',I,I'] * T[β,β',J,J'] * T[γ,γ',K,K'] * ρ[α',β',γ',I',J',K']
    return  (2/sqrt(π)*transtensor.wt[t]).*V
end


function poisson_equation!(V::AbstractArray{<:Any,6},
                          ρ::AbstractArray{<:Any,6},
                          transtensor::AbtractTransformationTensor,
                          t)
    T = transtensor[:,:,:,:,t]
    @tensoropt V[α,β,γ,I,J,K] = T[α,α',I,I'] * T[β,β',J,J'] * T[γ,γ',K,K'] * ρ[α',β',γ',I',J',K']
    return  (2/sqrt(π)*transtensor.wt[t]).*V
end


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

    V = ρ.*transtensor.wt[1]
    nt = size(transtensor)[end]
    @debug "nt=$nt"
    @debug "V type = $(typeof(V))"
    ptime = showprogress ? 0.3 : Inf
    if nworkers() > 1
        tmp = similar(V) # Yes it is a hack
        V = @showprogress ptime "Poisson equation... " @distributed (+) for t in 1:nt
            poisson_equation!(tmp, ρ, transtensor, t)
        end
    else
        tmp = similar(V)
        @showprogress ptime "Poisson equation... " for t in 1:nt
            V .+= poisson_equation!(tmp, ρ, transtensor[:,:,:,:,t], transtensor.wt[t])
        end
    end
    if tmax != nothing
        return V .+ coulomb_correction(ρ, tmax)
    end
    return V
end

function poisson_equation(ψ::QuantumState, transtensor::AbtractTransformationTensor;
                          tmax=nothing, showprogress=false)
    @assert dimension(ψ) == dimension(u"bohr^-2")
    ψ = uconvert(u"bohr^-2", ψ)
    V = poisson_equation(ψ.ψ, transtensor, tmax=tmax, showprogress=showprogress)
    return QuantumState(ψ.elementgrid, V)
end
