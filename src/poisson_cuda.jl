
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