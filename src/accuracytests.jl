using TensorOperations
using QuadGK


function gaussiandensity_self_energy(rtol=1e-12)
    _f(x) = (π/sqrt(2x^2+1))^3
    return quadgk(_f, 0, Inf, rtol=rtol)
end


"""
    test_accuracy(n_elements, n_gaussp, n_tpoints; Keywords) -> Float

Calculates self-energy for Gaussian change density numerically and compares it analytic result.

# Arguments
- `n_elements`  : Number of finite elements in calculation
- `n_gaussp`    : Number of Gauss points in element per degree of freedom
- `n_tpoins`    : Number of Gauss points for t-variable integration

# Keywords
- `tmax=20`          : Maximum value in t integration
- `tmin=0`           : Minimum value in t integration
- `correction=true`  : Add correction to electric field calculation
- `cmax=10`          : Limits for numerical calculation - cubic box from `-cmax` to `cmax`
- `mode=:normal`     : Integration method `:normal`,  `:normal_alt` or `:harrison`
- `δ=1.0`            : Area parameter to determine average value in `:normal_alt` δ∈]0,1]
- `μ=0`              : Parameter for `:harrison` mode
"""
function test_accuracy(n_elements, n_gaussp, n_tpoints;
                       tmax=20,
                       tmin=0,
                       correction=true,
                       cmax=10,
                       mode=:normal,
                       δ=1.0,
                       μ=0)
    @info "Initializing elements and Gauss points"
    eq = CubicElements(-cmax, cmax, n_elements)

    x, w = gausspoints(eq, n_gaussp)
    centers = getcenters(eq)

    @info "Generating transformation tensor"
    if mode in [:alt, :normal_alt]
        @info "Mode is normal with local mean values for T-tensor"
        T, t, wt = transformation_tensor_alt(eq, x, w, n_tpoints; δ=δ, tmin=tmin)
    elseif mode == :normal
        @info "Mode is normal"
        T, t, wt = transformation_tensor(centers, x, w, n_tpoints; tmax=tmax, tmin=tmin)
    elseif mode == :harrison
        @info "Mode is Harrison"
        T, t, wt = transformation_harrison(eq, x, w, n_tpoints; tmax=tmax, μ=μ)
    elseif mode == :harrison_alt
        @info "Mode is Harrison with local mean values for T-tensor"
        T, t, wt = transformation_harrison_alt(eq, x, w, n_tpoints; tmax=tmax, δ=δ, μ=μ)
    else
        @warn "Mode not recognised. Using normal mode"
        T, t, wt = transformation_tensor(centers, x, w, n_tpoints; tmax=tmax, tmin=tmin)
    end
    @info "Generating electron density tensor"


    ρ = density_tensor(centers, x)
    V = coulomb_tensor(ρ, T, x, w, t, wt)


    if correction
        V = V .+ (π/tmax^2).*ρ   # add correction
    end




    # Integraton weights fore elements + Gausspoints in tersor form
    ω = hcat([w for i in 1:n_elements]...)

    cc = V.*ρ
    E = @tensor ω[α,I]*ω[β,J]*ω[γ,K]*cc[α,β,γ,I,J,K]
    E_true = 21.92474849998632 # = gaussiandensity_self_energy()[1]
    @info "Calculated energy = $E"
    @info "True energy = $E_true"
    @info "Error = $(E-E_true) ; error/E = $( round((E-E_true)/E_true; sigdigits=1))"
    return E, E-E_true
end
