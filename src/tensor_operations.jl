using TensorOperations

function integrate(ϕ, grid::CubicElementGrid, ψ)
    # deprecated
    ω = ω_tensor(grid)
    c = ϕ.*ψ
    return  @tensor ω[1,2]*ω[3,4]*ω[5,6]*c[1,3,5,2,4,6]
end

function integrate(grid::CubicElementGrid, ρ)
    # deprecated
    ω = ω_tensor(grid)
    return  @tensor ω[1,2]*ω[3,4]*ω[5,6]*ρ[1,3,5,2,4,6]
end

function integrate(ωx::T, ωy::T, ωz::T, ρ::DT) where {T,DT}
    return @tensor ωx[1,2]*ωy[3,4]*ωz[5,6]*ρ[1,3,5,2,4,6]
end

function poisson_equation!(V::AbstractArray{<:Any,6},
                          ρ::Array{<:Any,6},
                          T::AbstractArray{<:Any,4},
                          wt)
    @assert size(V) == size(ρ)
    @tensor V[-1,-2,-3,-4,-5,-6] = T[-1,1,-4,2] * T[-2,3,-5,4] * T[-3,5,-6,6] * ρ[1,3,5,2,4,6]
    V .*= (2/sqrt(π)*wt)
    return V
end


function poisson_equation(ρ::Array{<:Any,6}, transtensor::AbtractTransformationTensor;
                          correction=true, showprogress=false)

    tmp = ρ.*transtensor.wt[1] # Make sure we have the correct type
    nt = size(transtensor, 5)
    @debug "nt=$nt"
    ptime = showprogress ? 1 : Inf
    p = Progress(nt, ptime)
    V = sum( axes(transtensor.wt, 1) ) do t
        poisson_equation!(tmp, ρ, transtensor[:,:,:,:,t], transtensor.wt[t])
        next!(p)
        tmp
    end
    if correction
        return V .+ coulomb_correction(ρ, transtensor.tmax)
    end
    return V
end

function poisson_equation(ψ::QuantumState{<:Array, <:Any}, transtensor::AbtractTransformationTensor;
    correction=true, showprogress=false)
#@assert dimension(ψ) == dimension(u"bohr^-2")
    ψ = auconvert(ψ)  # Length needs to be in bohr's 
    V = poisson_equation(ψ.psi, transtensor, correction=correction, showprogress=showprogress)
    return QuantumState(ψ.elementgrid, V, unit(ψ)*u"bohr^2")
end


## Operators

# Old version that is faster by a little
function (lo::LaplaceOperator)(qs::QuantumState{<:Array, 6})
    @tensor ϕ[i,j,k,I,J,K]:= qs.psi[ii,j,k,I,J,K]*lo.d2x[i,ii] + qs.psi[i,jj,k,I,J,K]*lo.d2y[j,jj] + qs.psi[i,j,kk,I,J,K]*lo.d2z[k,kk]
    return QuantumState(get_elementgrid(qs), ϕ, unit(qs)*unit(lo))
end

function operate_x!(dψ::AbstractArray{<:Any,6}, dt::DerivativeTensor, ψ::Array{<:Any,6})
    @tensor dψ[i,j,k,I,J,K] = dt.values[i,l] * ψ[l,j,k,I,J,K]
    return dψ
end

function operate_y!(dψ::AbstractArray{<:Any,6}, dt::DerivativeTensor, ψ::Array{<:Any,6})
    @tensor dψ[i,j,k,I,J,K] = dt.values[j,l] * ψ[i,l,k,I,J,K]
    return dψ
end

function operate_z!(dψ::AbstractArray{<:Any,6}, dt::DerivativeTensor, ψ::Array{<:Any,6})
    @tensor dψ[i,j,k,I,J,K] = dt.values[k,l] * ψ[i,j,l,I,J,K]
    return dψ
end