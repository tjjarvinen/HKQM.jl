using TensorOperations
using QuadGK
using Zygote
using ForwardDiff
using StaticArrays




function gaussiandensity_self_energy(rtol=1e-12; tmin=0, tmax=Inf)
    _f(x) = 1/(2x^2+1)^(3//2)
    return 2π^(5//2).*quadgk(_f, tmin, tmax, rtol=rtol)
end


function gaussian_coulomb_integral(a₁=1, a₂=1,  d=0; rtol=1e-12, tmin=0, tmax=Inf)
    @assert a₁>0
    @assert a₂>0
    function _f(t; a₁= a₁, a₂=a₂, d=d)
        c = (a₁+a₂)*t^2 +  a₁*a₂
        return exp(-( a₁*a₂/c)*t^2*d^2)/c^(3//2)
    end
    return 2π^(5//2).*quadgk(_f, tmin, tmax, rtol=rtol)
end

function gaussian_coulomb_integral_grad(a₁=1, a₂=1,  d=0; rtol=1e-12, tmin=0, tmax=Inf)
    function eri_ssss(x)
        a1 = x[1]
        a2 = x[2]
        d = x[3]
        boys(x; n=0) = quadgk( t-> t^(2n)*exp(-x[1]*t^2), 0, 1; rtol=1e-13)[1]
        θ = a1*a2/(a1+a2)
        F₀=boys(θ*d^2)
        return 2π^(5//2)*sqrt(θ)*F₀/(a1*a2)^(3//2)
    end
    h(x)=Zygote.gradient( z->Zygote.forwarddiff(eri_ssss, z), x)[1]
    return h( [Float64(a₁), Float64(a₂), Float64(d)])
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
- `tmax=300`          : Maximum t-value for integration
- `tmin=0`            : Minimum t-value for integration
- `mode=:combination` : Integration type - options `:normal`, `:log`, `loglocal`, `:local` and `:combination`
- `δ=0.25`            : Localization parameter for local integration types
- `correction=true`   : Correction to `tmax`-> ∞ integration - `true` calculate correction, `false` do not calculate
- `tboundary=20`      : Parameter for `:combination` mode. Switch to `:loglocal` for t>`tboundary` else use `:log`
- `α1=1`              : 1st Gaussian is exp(-α1*r^2)
- `α2=1`              : 2nd Gaussian is exp(-α2*r^2)
- `d=0`               : Distance between two Gaussian centers
"""
function test_accuracy(a::Real, ne::Int, ng::Int, nt::Int;
     tmax=300, tmin=0, mode=:combination, δ=0.25, correction=true, tboundary=20, α1=1, α2=1, d=0)
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
    test_accuracy(ceg, ct; correction=correction, α1=α1, α2=α2, d=d)
end

function test_accuracy(ceg::CubicElementGrid, ct::AbstractCoulombTransformation;
                            α1=1, α2=1, d=0, correction=true)
    tmin = ct.tmin
    tmax = ct.tmax
    r1 = SVector(0.5d, 0., 0.)
    r2 = SVector(-0.5d, 0., 0.)
    ρ1 = density_tensor(ceg; a=α1, r=r1)
    ρ2 = density_tensor(ceg; a=α2, r=r2)
    V = coulomb_tensor(ρ1, ct)

    E_cor = integrate(ρ2, ceg, coulomb_correction(ρ1, tmax))
    E_int = integrate(ρ2, ceg, V)
    E_tail = gaussian_coulomb_integral(α1, α2, d;tmin=tmax)[1]
    E_true = gaussian_coulomb_integral(α1, α2, d;tmin=tmin, tmax=tmax)[1]
    if correction
        E = E_int + E_cor
    else
        E = E_int
    end
    E_tot = gaussian_coulomb_integral(α1, α2, d)[1]
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

function test_accuracy_ad(a, ne, ng, nt; tmax=300, α1=1, α2=1, d=0, δ=0.25)
    ceg = CubicElementGrid(a, ne, ng)
    ct = loglocalct(ceg, nt; tmax=tmax, δ=δ, tboundary=20)
    return test_accuracy_ad(ceg::CubicElementGrid, ct::AbstractCoulombTransformation; α1=α1, α2=α2, d=d)
end


function test_accuracy_ad(ceg::CubicElementGrid, ct::AbstractCoulombTransformation; α1=1, α2=1, d=0)
    function f(x)
        tmin = ct.tmin
        tmax = ct.tmax
        r1 = SVector(0.5x[3], 0., 0.)
        r2 = SVector(-0.5x[3], 0., 0.)
        ρ1 = density_tensor(ceg; a=x[1], r=r1)
        ρ2 = density_tensor(ceg; a=x[2], r=r2)
        V = coulomb_tensor(Array(ρ1), Array(ct), Array(ct.wt); tmax=tmax)
        return integrate(ρ2, ceg, V)
    end
    x = [Float64(α1), Float64(α2), Float64(d)]
    @info "Zygote forward mode"
    g = Zygote.gradient( z->Zygote.forwarddiff(f, z), x)[1]
    #@info "ForwardDiff"
    #gf = x -> ForwardDiff.gradient(f, x)
    #fd = gf(x)
    g_ref = gaussian_coulomb_integral_grad(α1, α2, d)
    @info "Zygote forward mode gradient $g"
    #@info "ForwardDiff $fd"
    @info "Analytic $g_ref"
    @info "Relative error  $( round.((g.-g_ref)./g_ref; sigdigits=2))"
    return (g, g_ref)
end
