
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
    Tx::KernelTensor1D{T},
    Ty::KernelTensor1D{T},
    Tz::KernelTensor1D{T}
) where T
    fill!(out, zero(T))

    tmp_z = similar(ρ)
    tmp_y = similar(ρ)

    nt = length(Tx.nodes)

    @assert length(Ty.nodes) == nt
    @assert length(Tz.nodes) == nt

    for it in 1:nt
        w = Tx.weights[it]

        Kx = @view Tx.Tmat[:, :, it]
        Ky = @view Ty.Tmat[:, :, it]
        Kz = @view Tz.Tmat[:, :, it]

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
    Tx::KernelTensor1D{T},
    Ty::KernelTensor1D{T},
    Tz::KernelTensor1D{T},
    k::T,
) where T
    fill!(out, zero(T))

    tmp_z = similar(f)
    tmp_y = similar(f)

    nt = length(Tx.nodes)

    for it in 1:nt
        t = Tx.nodes[it]
        w = Tx.weights[it] * exp(-(k*k) / (T(4) * t*t))

        Kx = @view Tx.Tmat[:, :, it]
        Ky = @view Ty.Tmat[:, :, it]
        Kz = @view Tz.Tmat[:, :, it]

        @tensor tmp_z[a,b,c] := Kz[c,cp] * f[a,b,cp]
        @tensor tmp_y[a,b,c] := Ky[b,bp] * tmp_z[a,bp,c]
        @tensor out[a,b,c] += w * Kx[a,ap] * tmp_y[ap,b,c]
    end

    out .*= T(2) / sqrt(T(pi))

    return out
end