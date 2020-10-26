using TensorOperations
using QuadGK
using Zygote
using ForwardDiff




function gaussiandensity_self_energy(rtol=1e-12; tmin=0, tmax=Inf)
    _f(x) = (π/sqrt(2x^2+1))^3
    return quadgk(_f, tmin, tmax, rtol=rtol)
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
    eq = CubicElements(2cmax, n_elements)

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


    ρ = density_tensor_old(centers, x)
    V = coulomb_tensor(ρ, T, x, w, t, wt)


    if correction
        V = V .+ (π/tmax^2).*ρ   # add correction
    end




    # Integraton weights fore elements + Gausspoints in tersor form
    ω = hcat([Array(w) for i in 1:n_elements]...)

    cc = V.*ρ
    E = @tensor ω[α,I]*ω[β,J]*ω[γ,K]*cc[α,β,γ,I,J,K]
    E_true = 21.92474849998632 # = gaussiandensity_self_energy()[1]
    @info "Calculated energy = $E"
    @info "True energy = $E_true"
    @info "Error = $(E-E_true) ; error/E = $( round((E-E_true)/E_true; sigdigits=1))"
    return E, E-E_true
end


function test_accuracy_new(a::Real, ne::Int, ng::Int, nt::Int;
     tmax=25, tmin=0, mode=:normal, ae=1.0, δ=0.25, correction=true)
    ceg = CubicElementGrid(a, ne, ng)
    if mode == :normal
        @info "Normal mode"
        ct = CoulombTransformation(ceg, nt; tmax=tmax, tmin=tmin)
    elseif mode == :log
        @info "Logritmic mode"
        ct = CoulombTransformationLog(ceg, nt; tmax=tmax, tmin=tmin)
    elseif mode == :local
        @info "Local mode"
        ct = CoulombTransformationLocal(ceg, nt; tmax=tmax, tmin=tmin, δ=δ)
    elseif mode == :loglocal
        @info "Log local mode"
        ct = CoulombTransformationLogLocal(ceg, nt; tmax=tmax, tmin=tmin, δ=δ)
    else
        error("Mode not known")
    end
    test_accuracy_new(ceg, ct; correction=correction, ae=ae)
end

function test_accuracy_new(ceg::CubicElementGrid, ct::AbstractCoulombTransformation;
                            ae=1.0, correction=true)
    tmin = ct.tmin
    tmax = ct.tmax
    ρ = density_tensor(ceg; a=ae)
    V = coulomb_tensor(ρ, ct)

    E_cor = integrate(ρ, ceg, coulomb_correction(ρ, tmax))
    E_int = integrate(ρ, ceg, V)
    E_tail = gaussiandensity_self_energy(;tmin=tmax)[1]
    E_true = gaussiandensity_self_energy(;tmin=tmin, tmax=tmax)[1]
    if correction
        E = E_int + E_cor
    else
        E = E_int
    end
    E_tot = gaussiandensity_self_energy()[1]
    @info "Calculated energy = $E"
    @info "True inregration energy = $E_true"
    @info "Total Energy (reference) = $E_tot"
    @info "Integration error = $(round((E_int-E_true); sigdigits=2)) ; error/E = $( round((E_int-E_true)/E_true; sigdigits=2))"
    @info "Integration error relative to total energy = $( round((E_int-E_true)/E_tot; sigdigits=2))"
    @info "Error to total energy error/E_tot = $(round((E-E_tot)/E_tot; sigdigits=2))"
    @info "Tail energy = $(round(E_tail; sigdigits=2)) ; E_tail/E_tot = $(round(E_tail/E_tot; sigdigits=2))"
    @info "Energy correction = $(round(E_cor; sigdigits=2))  ; (E_cor-E_tail)/E = $(round((E_cor-E_tail)/E_tot;sigdigits=2))"
    return E
end

function test_accuracy_ad(a, ne, ng, nt; tmax=25, mode=:normal, ae=1., h=1e-3)
    function f(aa)
        ceg = CubicElementGrid(a, ne, ng)
        if mode == :normal
            @info "Normal mode"
            ct = CoulombTransformation(ceg, nt; tmax=tmax)
        elseif mode == :log
            @info "Logritmic mode"
            ct = CoulombTransformationLog(ceg, nt; tmax=tmax)
        else
            error("Mode not known")
        end
        ρ = density_tensor(ceg; a=aa[1])
        V = coulomb_tensor(ρ, Array(ct), Array(ct.wt))
        V = V .+ coulomb_correction(ρ, tmax)
        ω = ω_tensor(ceg)
        cc = V.*ρ
        E = @tensor ω[α,I]*ω[β,J]*ω[γ,K]*cc[α,β,γ,I,J,K]
        return E
    end
    g = x -> ForwardDiff.gradient(f, [x])[1];
    a2 = ae+h
    a1 = ae-h
    E1 = f(a1)
    E2 = f(a2)
    d_num = (E2-E1)/(a2-a1)
    @info "Numerical derivative=$d_num"
    @info "Zygote forward mode"
    zd_ad = Zygote.forwarddiff(f, ae)
    @info "Zygote = $zd_ad"
    @info "ForwardDiff"
    fd_ad = g(ae)
    @info "ForwardDiff = $fd_ad"
    @info "Difference between AD methods = $(zd_ad-fd_ad)"
    @info "Difference between ForwardDiff and numerical = $(fd_ad-d_num)"
    return fd_ad, d_num
end
