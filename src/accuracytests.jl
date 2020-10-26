using TensorOperations
using QuadGK
using Zygote
using ForwardDiff




function gaussiandensity_self_energy(rtol=1e-12; tmin=0, tmax=Inf)
    _f(x) = (π/sqrt(2x^2+1))^3
    return quadgk(_f, tmin, tmax, rtol=rtol)
end

"""
    test_accuracy(a::Real, ne::Int, ng::Int, nt::Int; kwords) -> Float64

Test accuracy on Gaussian charge distribution self energy.

# Arguments
- `a::Real`   : Simulation box size
- `ne::Int`   : Number of elements per dimension
- `ng::Int`   : Number of Gausspoints per dimension for r
- `nt::Int`   : Number of Gausspoints per dimension for t

# Keywords
- `tmax=25`      : Maximum t-value for integration
- `tmin=0`       : Minimum t-value for integration
- `mode=:normal` : Integration type - options `:normal`, `:log`, `loglocal`, `:local` and `:combination`
- `δ=0.25`       : Localization parameter for local integration types
- `correction=true`   : Correction to `tmax`-> ∞ integration - `true` calculate correction, `false` do not calculate
- `tboundary=20`      : Parameter for `:combination` mode. Switch to `:loglocal` for t>`tboundary` else use `:log`
"""
function test_accuracy(a::Real, ne::Int, ng::Int, nt::Int;
     tmax=25, tmin=0, mode=:normal, δ=0.25, correction=true, tboundary=20)
    ae=1.0
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
    elseif mode == :combination
        @info "Combination mode"
        ct = loglocalct(ceg, nt; tmax=tmax, δ=δ, tboundary=tboundary)
    else
        error("Mode not known")
    end
    test_accuracy(ceg, ct; correction=correction, ae=ae)
end

function test_accuracy(ceg::CubicElementGrid, ct::AbstractCoulombTransformation;
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
    @info "Tail energy = $(round(E_tail; sigdigits=5)) ; E_tail/E_tot = $(round(E_tail/E_tot; sigdigits=2))"
    @info "Energy correction = $(round(E_cor; sigdigits=5))  ; (E_cor-E_tail)/E_tot = $(round((E_cor-E_tail)/E_tot;sigdigits=2))"
    @info "Error to total energy error/E_tot = $(round((E-E_tot)/E_tot; sigdigits=2))"
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
