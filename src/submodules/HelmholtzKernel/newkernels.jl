"""
    FPrimitive{T}

Stores Fₖ function parameters:

- Polynomian coeffs for poly * Gaussian
- Constant for const
"""
struct FPrimitive{T}
    t::T
    gaussian_poly::Vector{T}
    erf_coeff::T
end


@inline function evaluate(F::FPrimitive{T}, x) where T
    xT = T(x)
    tx = F.t * xT
    erf_part = F.erf_coeff * erf(tx)

    if isempty(F.gaussian_poly) # we have F₀
        return erf_part
    else                        # Fᵢ i>0
        return evalpoly(xT, F.gaussian_poly) * exp(-(tx * tx)) + erf_part
    end
end


function build_F_primitives(maxk::Int, t::T) where {T}
    @assert maxk ≥ 0
    F = Vector{FPrimitive{T}}(undef, maxk + 1)

    F[1] = FPrimitive{T}(t, T[], sqrt(T(pi)) / (2t))          # F_0
    F[2] = FPrimitive{T}(t, T[-one(T)/(2t^2)], zero(T))       # F_1

    for k in 2:maxk
        scale = T(k - 1) / (2t^2)

        old = F[k-1]  # F_{k-2}, because Julia index k-1 corresponds to degree k-2

        poly = zeros(T, k)  # powers 0 through k-1

        for i in eachindex(old.gaussian_poly)
            poly[i] += scale * old.gaussian_poly[i]
        end

        poly[k] += -one(T) / (2t^2)  # z^(k-1)

        erf_coeff = scale * old.erf_coeff

        F[k+1] = FPrimitive{T}(t, poly, erf_coeff)
    end

    return F
end


@inline function evaluate_H(r::Int, F_next::FPrimitive{T}, z) where T
    r1 = T(r + 1)
    t = F_next.t
    zT = T(z)
    tz = t * zT

    return (zT^(r + 1) / r1) * erf(tz) -
           (2 * t / (sqrt(T(pi)) * r1)) * evaluate(F_next, zT)
end

function evaluate_E_all(s, c, d, Fs::Vector{FPrimitive{T}}) where T
    maxn = length(Fs) - 2
    Es = Vector{T}(undef, maxn + 1)

    sT = T(s)
    cT = T(c)
    dT = T(d)

    @inbounds for n in 0:maxn
        total = zero(T)

        for r in 0:n
            coeff = T(binomial(n, r)) * sT^(n-r)

            if isodd(r)
                coeff = -coeff
            end

            H_sc = evaluate_H(r, Fs[r+2], sT - cT)
            H_sd = evaluate_H(r, Fs[r+2], sT - dT)

            total += coeff * (H_sc - H_sd)
        end

        Es[n+1] = total
    end

    return Es
end


function evaluate_G_all(s, c, d, Fs::Vector{FPrimitive{T}}) where T
    maxF = length(Fs) - 1  # Fs[1] = F₀, Fs[maxF+1] = F_maxF

    Gs = zeros(T, maxF + 1, maxF + 1)

    sT = T(s)
    cT = T(c)
    dT = T(d)

    zc = sT - cT
    zd = sT - dT

    # ΔF[ℓ+1] = F_ℓ(s-c) - F_ℓ(s-d)
    ΔF = Vector{T}(undef, maxF + 1)

    @inbounds for ℓ in 0:maxF
        ΔF[ℓ + 1] = evaluate(Fs[ℓ + 1], zc) -
                    evaluate(Fs[ℓ + 1], zd)
    end

    @inbounds for n in 0:maxF
        for m in 0:maxF
            if n + m <= maxF
                total = zero(T)

                for r in 0:n
                    coeff = T(binomial(n, r)) * sT^(n - r)

                    if isodd(r)
                        coeff = -coeff
                    end

                    total += coeff * ΔF[m + r + 1]
                end

                Gs[n + 1, m + 1] = total
            end
        end
    end

    return Gs
end


function evaluate_I_all(
    Px::Int,
    Qy::Int,
    a,
    b,
    c,
    d,
    Fs::Vector{FPrimitive{T}},
) where T
    @assert Px ≥ 0
    @assert Qy ≥ 0

    # Need F up to F_{Px+Qy+1}
    @assert length(Fs) >= Px + Qy + 2

    aT = T(a)
    bT = T(b)
    cT = T(c)
    dT = T(d)

    Es_a = evaluate_E_all(aT, cT, dT, Fs)
    Es_b = evaluate_E_all(bT, cT, dT, Fs)

    Gs_a = evaluate_G_all(aT, cT, dT, Fs)
    Gs_b = evaluate_G_all(bT, cT, dT, Fs)

    Is = Matrix{T}(undef, Px + 1, Qy + 1)

    @inbounds for p in 0:Px
        for q in 0:Qy
            total = zero(T)

            for k in 0:p
                n = p + q - k
                coeff = T(binomial(p, k))

                J_b = integrate_shifted_F_from_tables(n, k, Fs, Es_b, Gs_b)
                J_a = integrate_shifted_F_from_tables(n, k, Fs, Es_a, Gs_a)

                total += coeff * (J_b - J_a)
            end

            Is[p + 1, q + 1] = total
        end
    end

    return Is
end

@inline function integrate_shifted_F_from_tables(
    n::Int,
    k::Int,
    Fs::Vector{FPrimitive{T}},
    Es::Vector{T},
    Gs::Matrix{T},
) where T
    Fk = Fs[k + 1]   # Fs[1] == F₀

    total = zero(T)

    # Gaussian polynomial part
    @inbounds for j in eachindex(Fk.gaussian_poly)
        m = j - 1
        total += Fk.gaussian_poly[j] * Gs[n + 1, m + 1]
    end

    # erf part
    if Fk.erf_coeff != zero(T)
        total += Fk.erf_coeff * Es[n + 1]
    end

    return total
end



function evaluate_K_block(
    elem_x,
    elem_y,
    t;
    precision::Int = 256,
    return_type = Float64,
)
    setprecision(BigFloat, precision) do
        coeffs_x = global_polynomial_coefficients_big(elem_x; precision=precision)
        coeffs_y = global_polynomial_coefficients_big(elem_y; precision=precision)

        Px = size(coeffs_x, 2) - 1
        Qy = size(coeffs_y, 2) - 1

        a = BigFloat(ustrip(elem_x.element.low))
        b = BigFloat(ustrip(elem_x.element.high))
        c = BigFloat(ustrip(elem_y.element.low))
        d = BigFloat(ustrip(elem_y.element.high))

        tB = BigFloat(t)

        Fs = build_F_primitives(Px + Qy + 1, tB)
        Is = evaluate_I_all(Px, Qy, a, b, c, d, Fs)

        Kbig = Matrix{BigFloat}(undef, size(coeffs_x, 1), size(coeffs_y, 1))

        @inbounds for i in axes(coeffs_x, 1)
            for j in axes(coeffs_y, 1)
                total = zero(BigFloat)

                for p in 0:Px
                    for q in 0:Qy
                        total += coeffs_x[i, p+1] *
                                 coeffs_y[j, q+1] *
                                 Is[p+1, q+1]
                    end
                end

                Kbig[i, j] = total
            end
        end

        return return_type.(Kbig)
    end
end


function evaluate_K_block(
    coeffs_x::AbstractMatrix{T},  # size nbasis_x × (Px+1)
    coeffs_y::AbstractMatrix{T},  # size nbasis_y × (Qy+1)
    I::AbstractMatrix{T},
) where T
    nbx = size(coeffs_x, 1)
    nby = size(coeffs_y, 1)

    K = Matrix{T}(undef, nbx, nby)

    @inbounds for i in 1:nbx
        for j in 1:nby
            total = zero(T)

            for p in 0:size(coeffs_x, 2)-1
                for q in 0:size(coeffs_y, 2)-1
                    total += coeffs_x[i, p+1] * coeffs_y[j, q+1] * I[p+1, q+1]
                end
            end

            K[i, j] = total
        end
    end

    return K
end



function local_polynomial_coefficients_big(eg; precision=256)
    setprecision(BigFloat, precision) do
        ξ = BigFloat.(eg.basis.nodes)
        N = length(ξ)

        V = Matrix{BigFloat}(undef, N, N)

        @inbounds for i in 1:N
            V[i, 1] = one(BigFloat)
            for p in 2:N
                V[i, p] = V[i, p-1] * ξ[i]
            end
        end

        C = V \ Matrix{BigFloat}(I, N, N)

        return C'  # rows = basis functions, columns = powers of ξ
    end
end

function global_polynomial_coefficients_big(eg; precision=256)
    Cξ = local_polynomial_coefficients_big(eg; precision=precision)

    setprecision(BigFloat, precision) do
        N = size(Cξ, 2)

        scale = BigFloat(ustrip(eg.scaling))
        shift = BigFloat(ustrip(eg.shift))

        Cx = zeros(BigFloat, size(Cξ))

        @inbounds for i in axes(Cξ, 1)
            for p in 0:N-1
                c = Cξ[i, p+1] / scale^p

                for r in 0:p
                    Cx[i, r+1] += c *
                                  BigFloat(binomial(p, r)) *
                                  (-shift)^(p-r)
                end
            end
        end

        return Cx
    end
end


function element_global_indices(
    egv::ElementGridVectorLegendre,
    ie::Int,
)
    return findall(pair -> pair.first == ie, egv.index)
end


const ElementGridVector1D = Union{
    ElementGridVectorLegendre,
    ElementGridVectorLobatto,
}



function evaluate_K_matrix(
    egv_x::ElementGridVector1D,
    egv_y::ElementGridVector1D,
    t;
    precision::Int = 256,
    return_type = Float64,
)
    K = zeros(return_type, length(egv_x), length(egv_y))

    @inbounds for ix in eachindex(egv_x.elements)
        elem_x = egv_x.elements[ix]
        gx = element_global_indices(egv_x, ix)

        for iy in eachindex(egv_y.elements)
            elem_y = egv_y.elements[iy]
            gy = element_global_indices(egv_y, iy)

            Kblock = evaluate_K_block(
                elem_x,
                elem_y,
                t;
                precision = precision,
                return_type = return_type,
            )

            for i_local in axes(Kblock, 1)
                i_global = gx[i_local]

                for j_local in axes(Kblock, 2)
                    j_global = gy[j_local]

                    K[i_global, j_global] += Kblock[i_local, j_local]
                end
            end
        end
    end

    return K
end


function element_global_indices(
    egv::ElementGridVectorLobatto,
    ie::Int,
)
    elem = egv.elements[ie]
    nloc = length(elem)

    inds = Vector{Int}(undef, nloc)

    @inbounds for ig in 1:nloc
        idx = findfirst(==(ie => ig), egv.index)

        if idx === nothing
            # For Lobatto, the last point of element ie is shared with
            # the first point of element ie+1.
            if ig == nloc && ie < length(egv.elements)
                idx = findfirst(==((ie + 1) => 1), egv.index)
            else
                error("Could not map local index ($ie, $ig) to global index")
            end
        end

        inds[ig] = idx
    end

    return inds
end