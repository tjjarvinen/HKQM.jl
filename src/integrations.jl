
"""
    integrate(ϕ, grid::AbstractElementGridSymmetricBox, ψ)
    integrate(grid::AbstractElementGridSymmetricBox, ρ)
    integrate(ϕ::QuantumState, ψ::QuantumState)
    integrate(ωx::T, ωy::T, ωz::T, ρ::DT) where {T,DT}

Low lever integration routines. (users should not use these, as they can change)
"""
function integrate(ωx, ωy, ωz, ρ)
    tmp = permutedims(ρ, [1,4,2,5,3,6])
    s = size(tmp)
    rtmp = reshape(tmp, s[1]*s[2], prod( s[3:end] ) )
    rωx = reshape(ωx, 1, prod( size(ωx) ) )
    rωy = reshape(ωy, 1, prod( size(ωy) ) )
    rωz = reshape(ωz, prod( size(ωz) ) )

    tyz = rωx * rtmp  # integrate x-axis
    rtyz = reshape(tyz, s[3]*s[4], s[5]*s[6])
    tz = rωy * rtyz   # integrate y-axis

    return sum( tz * rωz )  # integrate z-axis  ## dot wont work with Metal.jl
end



"""
    bracket(ϕ::QuantumState, ψ::QuantumState)
    bracket(ϕ::QuantumState, op::AbstractOperator, ψ::QuantumState)

Equivalent to Dirac bracket notation <ϕ|ψ> and <ϕ|op|ψ>.
Returns expectation value.
"""
function bracket(ϕ::QuantumState{<:Array, <:Any}, ψ::QuantumState{<:Array, <:Any})
    # No problems with this version
    @assert size(ϕ) == size(ψ)
    eg = get_elementgrid(ψ)
    ωx = getweight(eg, 1)
    ωy = getweight(eg, 2)
    ωz = getweight(eg, 3)
    return integrate(ωx, ωy, ωz, ϕ ⋆ ψ)*unit(ϕ)*unit(ψ)
end

function bracket(ϕ::QuantumState, ψ::QuantumState)
    # We can have gpu arrays
    @assert size(ϕ) == size(ψ)
    eg = get_elementgrid(ψ) 
    ωx = getweight(eg, 1)
    ωy = getweight(eg, 2)
    ωz = getweight(eg, 3)
    nωx = similar(ψ.psi, eltype(ωx), size(ωx))
    copy!(nωx, ωx)
    nωy = similar(ψ.psi, eltype(ωy), size(ωy))
    copy!(nωy, ωy)
    nωz = similar(ψ.psi, eltype(ωz), size(ωz))
    copy!(nωz, ωz)

    return integrate(nωx, nωy, nωz, ϕ ⋆ ψ)*unit(ϕ)*unit(ψ)
end

"""
    bracket(T, ϕ::QuantumState, ψ::QuantumState)
    bracket(T, ϕ::QuantumState, op::AbstractOperator{1}, ψ::QuantumState)

Calculate Dirac bracket with given `T` array type.
This can be used to offload the calculation to GPU
by giving a GPU array type. 
"""
function bracket(T, ϕ::QuantumState, ψ::QuantumState)
    @assert size(ϕ) == size(ψ)
    eg = get_elementgrid(ψ)
    ωx = T( getweight(eg, 1) )
    ωy = T( getweight(eg, 2) )
    ωz = T( getweight(eg, 3) )

    Tϕ = convert_array_type(T, ϕ)
    Tψ = convert_array_type(T, ψ)
    return integrate(ωx, ωy, ωz, Tϕ ⋆ Tψ)*unit(ϕ)*unit(ψ)
end

function bracket(ϕ::QuantumState, op::AbstractOperator{1}, ψ::QuantumState)
    @assert size(ϕ) == size(ψ) == size(op)
    return bracket(ϕ, op*ψ)
end

function bracket(T, ϕ::QuantumState, op::AbstractOperator{1}, ψ::QuantumState)
    @assert size(ϕ) == size(ψ) == size(op)
    return bracket(T, ϕ, op*ψ)
end

function bracket(ϕ::QuantumState, op::AbstractOperator, ψ::QuantumState)
    @assert size(ϕ) == size(ψ) == size(op)
    return map( O->bracket(ϕ, O, ψ),  op)
end

function bracket(T, ϕ::QuantumState, op::AbstractOperator, ψ::QuantumState)
    #TODO make this more efficient. It now moves same data several times.
    @assert size(ϕ) == size(ψ) == size(op)
    return map( O->bracket(T, ϕ, O, ψ),  op)
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
    j = map( p ) do pᵢ
        tmp = real.(ψ⋆(pᵢ*ψ) .+ ϕ⋆(pᵢ*ϕ))
        QuantumState(get_elementgrid(ψ), tmp, unit(p)*unit(ψ))
    end
    return j
end

## Coulomb integral / Poisson equation

function coulomb_correction(ρ, tmax)
    T = eltype(ρ)
    return T( 2/sqrt(π)*(π/tmax^2) ) .* ρ
end

function poisson_equation!(
        V1::AbstractArray{<:Any,3},
        V2::AbstractArray{<:Any,3},
        ρ::AbstractArray{<:Any,3},
        Tx::AbstractArray{<:Any,2},
        Ty::AbstractArray{<:Any,2},
        Tz::AbstractArray{<:Any,2},
        wt
    )
    @assert length(V1) == length(V2) == length(ρ)
    # V1 and V2 work as temporary arrays to store
    # the mid calculation results.

    # Note that Tx, Ty and Tz are not symmetric due to
    # integration constant only being on the right side!
    # Thus we need to contract from the right side.  

    s = size(ρ)

    # Contract x-axis
    rxρ = reshape(ρ, s[1], s[2]*s[3])    # rxρ[x,yz]
    tmp1 = reshape(V1, s[1], s[2]*s[3])  # tmp1[x,yz]
    mul!(tmp1, Tx, rxρ)  # tmp1[x,yz] = Tx[x,k]*rxρ[k,yz]
    
    # Contract y-axis
    rtmp1 = reshape(tmp1, s[1], s[2], s[3])  # rtmp1[x,y,z]
    tmp2 = reshape(V2, s[2], s[1], s[3])     # tmp2[y,x,z]
    permutedims!(tmp2, rtmp1, [2,1,3])       # tmp2[y,x,z]
    rtmp2 = reshape(tmp2, s[2], s[1]*s[3])   # rtmp2[y,xz]
    tmp1 = reshape(V1, s[2], s[1]*s[3])      # tmp1[y,xz]
    mul!(tmp1, Ty, rtmp2) # tmp1[y,xz] = Ty[y,k]*rtmp2[k,xz]

    # Contract z-axis
    rtmp1 = reshape(tmp1, s[2], s[1], s[3])  # rtmp1[y,x,z]
    tmp2 = reshape(V2, s[3], s[1], s[2])     # tmp2[z,x,y]
    permutedims!(tmp2, rtmp1, [3,2,1])       # tmp2[z,x,y]
    rtmp2 = reshape(tmp2, s[3], s[1]*s[2])   # rtmp2[z,xy]
    tmp1 = reshape(V1, s[3], s[1]*s[2])      # tmp1[z,xy]
    mul!(tmp1, Tz, rtmp2) # tmp1[z,xy] = Tz[z,k]*rtmp2[k,xy]


    # Restore the correct permutation and shape
    rtmp1 = reshape(tmp1, s[3], s[1], s[2]) # rtmp1[z,x,y]
    permutedims!(V2, rtmp1, [2,3,1])        # V2[x,y,z]

    V2 .*= (2wt/sqrt(one(wt)*π))
    return V2
end


function poisson_equation(T, ρ::AbstractArray{<:Any,6}, transtensor::AbtractTransformationTensor;
        correction=true, showprogress=false)
    # Reshape and permute arrays to better suit contractions
    s = size(ρ)
    tmp = permutedims(ρ, [1,4,2,5,3,6])
    rtmp = reshape(tmp, s[1]*s[4], s[2]*s[5], s[3]*s[6])
    ttmp = give_whole_tensor(T, transtensor) #TODO does not count type of transtensor
    pttmp = permutedims(ttmp, [1,3,2,4,5])
    st = size(transtensor)
    T_tensor = reshape(pttmp, st[1]*st[3], st[2]*st[4], st[5])
    t_step = prod(st[1:4])  # step size to help copy data later

    # Create temporary arrays
    ET = promote_type(eltype(ρ), T, eltype(ttmp))  
    tmp1 = similar(rtmp, ET)
    tmp2 = similar(rtmp, ET)
    tensor = similar(ρ, T, st[1]*st[3], st[2]*st[4] )

    # Set up progress meter
    nt = size(transtensor, 5)
    @debug "nt=$nt"
    ptime = showprogress ? 1 : Inf
    p = Progress(nt, ptime)


    # Calculate
    V = sum( axes(transtensor.wt, 1) ) do t
        # copyto!(tensor, T_tensor[:,:,t]) but without extra allocation
        copyto!(tensor, 1, T_tensor, 1+(t-1)*t_step, t_step)
        tmp = poisson_equation!(tmp1, tmp2, rtmp, tensor, tensor, tensor, T(transtensor.wt[t]))
        next!(p)
        tmp
    end
    # Reshape V back to the correct shape and permutation
    s = size(ρ)
    rV = reshape(V, s[1], s[4], s[2], s[5], s[3], s[6])
    V_out = reshape(tmp, s...)
    permutedims!(V_out, rV, [1,3,5,2,4,6])
    if correction
        return V_out .+ coulomb_correction(ρ, transtensor.tmax)
    end
    return V_out
end

function poisson_equation(ρ::AbstractArray{<:Any,6}, transtensor::AbtractTransformationTensor;
    correction=true, showprogress=false)
    T = eltype(ρ)
    return poisson_equation(T, ρ, transtensor; correction=correction, showprogress=showprogress)
end

function poisson_equation(ψ::QuantumState, transtensor::AbtractTransformationTensor;
        correction=true, showprogress=false)
    ψ = auconvert(ψ)  # Length needs to be in bohr's 
    T = (eltype ∘ eltype ∘ get_elementgrid)(ψ)
    V = poisson_equation(T, ψ.psi, transtensor, correction=correction, showprogress=showprogress)
    return QuantumState(get_elementgrid(ψ), V, unit(ψ)*u"bohr^2")
end

function poisson_equation(ψ::QuantumState; showprogress=false)
    ct = optimal_coulomb_tranformation(ψ)
    return poisson_equation(ψ, ct ; showprogress=showprogress)
end
