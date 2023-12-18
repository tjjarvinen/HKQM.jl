
## Integral

function integrate(
    ega::AbstractElementGrid{T, 3}, 
    data::AbstractArray{Tt, 3}
) where {T, Tt}
    @assert size(ega) == size(data)

    tmp_x_yz = reshape(data, size(data,1), size(data,2)*size(data,3))
    ωx = get_weight(ega, 1)
    tmp_yz = ωx' * tmp_x_yz

    tmp_y_z = reshape(tmp_yz, size(data,2), size(data,3))
    ωy = get_weight(ega, 2)
    tmp_z = ωy' * tmp_y_z

    ωz = get_weight(ega, 3)
    return sum( ωz .* tmp_z' )
end


## Derivatives

function derivative_x!(
    out::AbstractArray{Tt, 3},
    ega::AbstractElementGrid{T, 3},
    data::AbstractArray{Tt, 3};
    order::Integer=1
) where {T, Tt}
    @assert size(ega) == size(data) == size(out)

    tmp_x_yz = reshape(data, size(data,1), size(data,2)*size(data,3))
    out_x_yz = reshape(out, size(data,1), size(data,2)*size(data,3))
    dx = get_derivative_matrix(ega, 1, order)
    mul!(out_x_yz, dx, tmp_x_yz)
    return out
end


function derivative_x(
    ega::AbstractElementGrid{T, 3},
    data::AbstractArray{Tt, 3};
    order::Integer=1
) where {T, Tt}
    out = similar(data)
    return derivative_x!(out, ega, data; order=order)
end


function derivative_y!(
    tmp::AbstractArray{Tt, 3},
    out::AbstractArray{Tt, 3},
    ega::AbstractElementGrid{T, 3},
    data::AbstractArray{Tt, 3};
    order::Integer=1
) where {T, Tt}
    @assert size(ega) == size(data) == size(out) == size(tmp)

    out_y_x_z = reshape(out, size(data,2), size(data,1), size(data,3))
    permutedims!(out_y_x_z, data, (2,1,3))
    out_y_xz  = reshape(out_y_x_z, size(data,2), size(data,1)*size(data,3))
    tmp_y_xz  = reshape(tmp, size(data,2), size(data,1)*size(data,3))
    dy = get_derivative_matrix(ega, 2, order)
    mul!(tmp_y_xz, dy, out_y_xz)
    tmp_y_x_z = reshape(tmp_y_xz, size(data,2), size(data,1), size(data,3))
    permutedims!(out, tmp_y_x_z, (2,1,3))
    return out
end

function derivative_y(
    ega::AbstractElementGrid{T, 3},
    data::AbstractArray{Tt, 3};
    order::Integer=1
) where {T, Tt}
    out = similar(data)
    tmp = similar(data)
    return derivative_y!(tmp, out, ega, data; order=order)
end


function derivative_z!(
    out::AbstractArray{Tt, 3},
    ega::AbstractElementGrid{T, 3},
    data::AbstractArray{Tt, 3};
    order::Integer=1
) where {T, Tt}
    @assert size(ega) == size(data) == size(out)

    tmp_xy_z = reshape(data, size(data,1)*size(data,2), size(data,3))
    out_xy_z = reshape(out, size(data,1)*size(data,2), size(data,3))
    dz = get_derivative_matrix(ega, 3, order)
    mul!(out_xy_z, tmp_xy_z, dz')
    return out
end


function derivative_z(
    ega::AbstractElementGrid{T, 3},
    data::AbstractArray{Tt, 3};
    order::Integer=1
) where {T, Tt}
    out = similar(data)
    return derivative_z!(out, ega, data; order=order)
end


function laplacian!(
    tmp::AbstractArray{Tt, 3},
    out::AbstractArray{Tt, 3},
    ega::AbstractElementGrid{T, 3},
    data::AbstractArray{Tt, 3}
) where {Tt, T}
    @assert size(ega) == size(data) == size(out) == size(tmp)
    derivative_y!(tmp, out, ega, data; order=2)
    derivative_x!(tmp, ega, data; order=2)
    out += tmp
    derivative_z!(tmp, ega, data; order=2)
    out += tmp
    return out
end


function laplacian(
    ega::AbstractElementGrid{T, 3}, 
    data::AbstractArray{Tt, 3}
) where {Tt, T}
    tmp = similar(data)
    out = similar(data)
    laplacian!(tmp, out, ega, data)
    return out
end


##

function apply_transformation(
    data::AbstractArray{Ta,3},
    tx::Tr,
    ty::Tr,
    tz::Tr;
    correction=true
)   where {Ta, Tr<:AbstractTransformationTensor}
    @assert size(tx,3) == size(ty,3) == size(tz,3)
    @assert size(data, 1) == size(tx, 2)
    @assert size(data, 2) == size(ty, 2)
    @assert size(data, 3) == size(tz, 2)
    T = Base.promote_eltype(data, tx, ty, tz)
    T = promote_type(T, typeof(get_t_weight(tx,1)))
    # note V1 and V2 are named after literature
    # and used for temporary arrays
    V1 = similar(data, T) 
    V2 = similar(data, T)


    V = sum( 1:size(tx,3) ) do tᵢ
        tmp = calculate_transformation(
            V1,
            V2,
            data,
            get_matrix_at_t(tx, tᵢ),
            get_matrix_at_t(ty, tᵢ),
            get_matrix_at_t(tz, tᵢ),
            get_t_weight(tx,tᵢ)
        )
        tmp
    end

    if correction
        tmax = get_t_max(tx)
        V2 .= T( 2/sqrt(π)*(π/tmax^2) ) .* data
        V .+= V2
    end
    return V
end


function calculate_transformation(
   V1::AbstractArray{T1,3},
   V2::AbstractArray{T1,3},
   data::AbstractArray{T,3},
   tx::AbstractMatrix{Tx},
   ty::AbstractMatrix{Ty},
   tz::AbstractMatrix{Tz},
   wt 
)   where {T, T1, Tx, Ty, Tz}
    @assert size(tx,3) == size(ty,3) == size(tz,3)
    @assert size(data, 1) == size(tx, 2) == size(tx, 1)
    @assert size(data, 2) == size(ty, 2) == size(ty, 1)
    @assert size(data, 3) == size(tz, 2) == size(tz, 1)
    @assert size(data) == size(V1) == size(V2)
    
    s = size(data)
    # Contract x-axis
    tmp = reshape(data, s[1], s[2]*s[3])
    V1_x_yz = reshape(V1, s[1], s[2]*s[3])
    mul!(V1_x_yz, tx, tmp)

    # Contract z-axis
    V1_xy_z = reshape(V1_x_yz, s[1]*s[2], s[3])
    V2_xy_z = reshape(V2, s[1]*s[2], s[3])
    mul!(V2_xy_z, V1_xy_z, tz')

    # Contract y-axis
    V1_y_x_z = reshape(V1, s[2], s[1], s[3])
    permutedims!(V1_y_x_z, V2, (2,1,3))
    V1_y_xz = reshape(V1_y_x_z, s[2], s[1]*s[3])
    V2_y_xz = reshape(V2, s[2], s[1]*s[3])
    mul!(V2_y_xz, ty, V1_y_xz)

    # Restore original permutation
    V2_y_x_z = reshape(V2_y_xz, s[2], s[1], s[3])
    permutedims!(V1, V2_y_x_z, (2,1,3))

    # Add integration weight
    V1 .*= convert( eltype(V1), (2wt/sqrt(one(wt)*π)) )
    return V1
end