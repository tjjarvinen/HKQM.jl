
function build_kernel_tensor_1d(
    grid::ElementGridVector1D,
    tgrid;
    precision::Int = 256,
    return_type = Float64,
    ntasks = Threads.nthreads(),
)
    nt = length(tgrid)
    n = length(grid)

    nodes = return_type.(ustrip.([tgrid[it] for it in 1:nt]))
    weights = return_type.(ustrip.(get_weight(tgrid)))

    Tmat = Array{return_type, 3}(undef, n, n, nt)

    @tasks for it in 1:nt
        @set ntasks = ntasks

        K = evaluate_K_matrix(
            grid,
            grid,
            nodes[it];
            precision = precision,
            return_type = return_type,
        )

        @views Tmat[:, :, it] .= K
    end

    return KernelTensor1D(nodes, weights, Tmat)
end


##


function apply_poisson!(
    out::Array{T,3},
    ρ::Array{T,3},
    K::KernelTensor3D{T},
) where T
    fill!(out, zero(T))

    tmp_z = similar(ρ)
    tmp_y = similar(ρ)

    nt = length(K.x.nodes)

    @assert length(K.y.nodes) == nt
    @assert length(K.z.nodes) == nt

    for it in 1:nt
        w = K.x.weights[it]

        Kx = @view K.x.Tmat[:, :, it]
        Ky = @view K.y.Tmat[:, :, it]
        Kz = @view K.z.Tmat[:, :, it]

        @tensor tmp_z[a,b,c] := Kz[c,cp] * ρ[a,b,cp]
        @tensor tmp_y[a,b,c] := Ky[b,bp] * tmp_z[a,bp,c]
        @tensor out[a,b,c] += w * Kx[a,ap] * tmp_y[ap,b,c]
    end

    out .*= T(2) / sqrt(T(pi))

    return out
end



function apply_helmholtz!(
    out::Array{T,3},
    f::Array{T,3},
    K::KernelTensor3D{T},
    k::T,
) where T
    fill!(out, zero(T))

    tmp_z = similar(f)
    tmp_y = similar(f)

    nt = length(K.x.nodes)

    for it in 1:nt
        t = K.x.nodes[it]
        w = K.x.weights[it] * exp(-(k*k) / (T(4) * t*t))

        Kx = @view K.x.Tmat[:, :, it]
        Ky = @view K.y.Tmat[:, :, it]
        Kz = @view K.z.Tmat[:, :, it]

        @tensor tmp_z[a,b,c] := Kz[c,cp] * f[a,b,cp]
        @tensor tmp_y[a,b,c] := Ky[b,bp] * tmp_z[a,bp,c]
        @tensor out[a,b,c] += w * Kx[a,ap] * tmp_y[ap,b,c]
    end

    out .*= T(2) / sqrt(T(pi))

    return out
end