
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
    return sum( ωz * tmp_z )
end


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