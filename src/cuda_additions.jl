
const CuArray = CUDA.CuArray


function poisson_equation!(V::CuArray{<:Any,6},
        ρ::CuArray{<:Any,6},
        T::CuArray{<:Any,4},
        wt)
    @assert size(V) == size(ρ)
    @cutensor V[-1,-2,-3,-4,-5,-6] = T[-1,1,-4,2] * T[-2,3,-5,4] * T[-3,5,-6,6] * ρ[1,3,5,2,4,6]
    V .*= (2/sqrt(π)*wt)
    return V
end


function poisson_equation(ρ::Array, transtensor::AbtractTransformationTensor;
        tmax=nothing, showprogress=false)
    @debug "GPU calculation"
    cu_ρ = CuArray(ρ)
    tmp = cu_ρ .* transtensor.wt[1] # Make sure we have correct type
    nt = size(transtensor, 5)
    @debug "nt=$nt"
    ptime = showprogress ? 1 : Inf
    p = Progress(nt, ptime)
    V = sum( axes(transtensor.wt, 1) ) do t
        poisson_equation!(tmp, cu_ρ, CuArray(transtensor[:,:,:,:,t]), transtensor.wt[t])
        next!(p)
        tmp
    end
    if tmax !== nothing
        return Array( V .+ coulomb_correction(cu_ρ, tmax) )
    end
    return Array(V)
end

function poisson_equation(ρ::CuArray, transtensor::AbtractTransformationTensor;
        tmax=nothing, showprogress=false)
    @debug "GPU calculation"
    tmp = ρ .* transtensor.wt[1] # Make sure we have correct type
    nt = size(transtensor, 5)
    @debug "nt=$nt"
    ptime = showprogress ? 1 : Inf
    p = Progress(nt, ptime)
    V = sum( axes(transtensor.wt, 1) ) do t
        poisson_equation!(tmp, ρ, CuArray(transtensor[:,:,:,:,t]), transtensor.wt[t])
        next!(p)
        tmp
    end
    if tmax !== nothing
        return V .+ coulomb_correction(ρ, tmax)
    end
    return V
end


##

function (lo::LaplaceOperator)(qs::QuantumState{<:Array, <:Any})
    @debug "GPU Laplace"
    tmp = CuArray(lo.g.dt)
    @cutensor w[i,j]:=tmp[i,k]*tmp[k,j]
    @cutensor ϕ[i,j,k,I,J,K]:= qs.psi[ii,j,k,I,J,K]*w[i,ii] + qs.psi[i,jj,k,I,J,K]*w[j,jj] + qs.psi[i,j,kk,I,J,K]*w[k,kk]
    return QuantumState(get_elementgrid(qs), Array(ϕ), unit(qs)*unit(lo.g)^2)
end

function (lo::LaplaceOperator)(qs::QuantumState{<:CuArray, <:Any})
    @debug "GPU Laplace"
    tmp = CuArray(lo.g.dt)
    @cutensor w[i,j]:=tmp[i,k]*tmp[k,j]
    @cutensor ϕ[i,j,k,I,J,K]:= qs.psi[ii,j,k,I,J,K]*w[i,ii] + qs.psi[i,jj,k,I,J,K]*w[j,jj] + qs.psi[i,j,kk,I,J,K]*w[k,kk]
    return QuantumState(get_elementgrid(qs), ϕ, unit(qs)*unit(lo.g)^2)
end