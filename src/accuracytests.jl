using TensorOperations
using QuadGK
using Zygote
using ForwardDiff
using StaticArrays
using LinearAlgebra




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


function gaussian_density_nuclear_potential(a=1, d=0; rtol=1e-12, tmin=0, tmax=Inf)
    #f(t) = exp(-a*t^2*d^2/(t^2+a)) * (t^2+a)^(-3//2)
    f(t) = (t^2+a)^(-3//2)
    return 2π.*quadgk(f, tmin, tmax; rtol=rtol)
end

"""
    test_nuclear_potential(a, ne::Int, ng::Int, nt::Int; Kwargs...) -> Dict

Test nuclear potential accuracy to Gaussien electron density.

# Arguments
- `a`        : Grid box size in bohr or unitful unit if given
- `ne::Int`  : Number of elements per dimension
- `ng::Int`  : Number of Gauss points per element per dimension
- `nt::Int`  : Number of Gauss points for t-integration

# Keywords
- `α=1`           :  Width of Gaussian
- `origin=0.`     :  Nuclear coordinate -> (origin, origin, origin)
- `tmin=0`        :  Minimum t-value
- `tmax=30`       :  Maximum t-value
- `δ=0.25`        :  Local correction parameter, if used
- `mode="normal"` :  Sets mode, options are: "normal", "log", "loglocal", "preset"
"""
function test_nuclear_potential(a, ne::Int, ng::Int, nt::Int;
                                α=1, σ=0.1, origin=zeros(3), tmin=0, tmax=30, δ=0.25, mode="normal")
    ceg = CubicElementGrid(a, ne, ng)
    if mode == "normal"
        @info "normal mode"
        npt1 = NuclearPotentialTensor(origin[1], ceg, nt; tmin=tmin, tmax=tmax)
        npt2 = NuclearPotentialTensor(origin[2], ceg, nt; tmin=tmin, tmax=tmax)
        npt3 = NuclearPotentialTensor(origin[3], ceg, nt; tmin=tmin, tmax=tmax)
    elseif mode == "log"
        @info "logarithmic mode"
        npt1 = NuclearPotentialTensorLog(origin[1], ceg, nt; tmin=tmin, tmax=tmax)
        npt2 = NuclearPotentialTensorLog(origin[2], ceg, nt; tmin=tmin, tmax=tmax)
        npt3 = NuclearPotentialTensorLog(origin[3], ceg, nt; tmin=tmin, tmax=tmax)
    elseif mode == "loglocal"
        @info "logarithmic mode with local correction δ=$δ"
        npt1 = NuclearPotentialTensorLogLocal(origin[1], ceg, nt; tmin=tmin, tmax=tmax, δ=δ)
        npt2 = NuclearPotentialTensorLogLocal(origin[2], ceg, nt; tmin=tmin, tmax=tmax, δ=δ)
        npt3 = NuclearPotentialTensorLogLocal(origin[3], ceg, nt; tmin=tmin, tmax=tmax, δ=δ)
    elseif mode == "preset"
        @info "Preset mode"
        v1 = optimal_nuclear_tensor(ceg, origin[1])
        v2 = optimal_nuclear_tensor(ceg, origin[2])
        v3 = optimal_nuclear_tensor(ceg, origin[3])
        tmin = v1.tmin
        tmax = v1.tmax
        @info "tmin is set to $tmin"
        @info "tmax is set to $tmax"
        vv = nuclear_potential(ceg, 1, v1,v2,v3)
        V = vv.vals
    elseif mode == "gaussian"
        @info "Gaussian mode"
        npt1 = NuclearPotentialTensorGaussian(origin[1], ceg, nt; tmin=tmin, tmax=tmax, σ=σ)
        npt2 = NuclearPotentialTensorGaussian(origin[2], ceg, nt; tmin=tmin, tmax=tmax, σ=σ)
        npt3 = NuclearPotentialTensorGaussian(origin[3], ceg, nt; tmin=tmin, tmax=tmax, σ=σ)
    else
        error("mode not recognized")
    end
    if mode != "preset"
        pt = PotentialTensor(npt1, npt2, npt3)
        V = Array(pt)
    end
    @info "Using using cubic box of size ($a)^3 = $(a^3)"
    @info "Using $ne^3 = $(ne^3) elements"
    @info "Using $ng^3 = $(ng^3) Gauss points per element"
    @info "Total ammount of points is $((ne*ng)^3)"
    @info "t-integration is using $nt points"
    r = position_operator(ceg)
    r0 = r - austrip.(collect(origin)).*u"bohr"
    r2 = r0⋅r0
    ρ = exp(-1u"bohr^-2"*α*r2).vals
    integral = integrate(ρ, ceg, V)
    ref = gaussian_density_nuclear_potential(α; tmin=tmin, tmax=tmax)
    ref_tot = gaussian_density_nuclear_potential(α)
    tail = gaussian_density_nuclear_potential(α; tmin=tmax)
    @info "Integral = $(integral)"
    @info "Reference = $(ref[1])"
    @info "Relative error = $( round((integral-ref[1])/ref[1]; sigdigits=2) )"
    @info "Total reference = $(ref_tot[1])"
    @info "Relative error to total = $( round((integral-ref[1])/ref_tot[1]; sigdigits=2) )"
    @info "Tail energy = $(round(tail[1]; sigdigits=2))"
    @info "Tail relative to total = $(round(tail[1]/ref_tot[1]; sigdigits=2))"
    return Dict("integral"=>integral,
                "reference"=>ref[1],
                "total reference"=>ref_tot[1],
                "tail"=>tail[1])
end



"""
    test_accuracy(a::Real, ne::Int, ng::Int, nt::Int; kwords) -> Dict

Test accuracy on Gaussian charge distribution self energy.

# Arguments
- `a::Real`   : Simulation box size
- `ne::Int`   : Number of elements per dimension
- `ng::Int`   : Number of Gausspoints per dimension for r
- `nt::Int`   : Number of Gausspoints per dimension for t

# Keywords
- `tmax=300`          : Maximum t-value for integration
- `tmin=0`            : Minimum t-value for integration
- `mode=:combination` : Integration type - options `:normal`, `:log`, `:loglocal`, `:local`, `:combination` and `:optimal`
- `δ=0.25`            : Localization parameter for local integration types
- `correction=true`   : Correction to `tmax`-> ∞ integration - `true` calculate correction, `false` do not calculate
- `tboundary=20`      : Parameter for `:combination` mode. Switch to `:loglocal` for t>`tboundary` else use `:log`
- `α1=1`              : 1st Gaussian is exp(-α1*r^2)
- `α2=1`              : 2nd Gaussian is exp(-α2*r^2)
- `d=0`               : Distance between two Gaussian centers
"""
function test_accuracy(a, ne::Int, ng::Int, nt::Int;
        tmax=300, tmin=0, mode=:optimal, δ=0.25, correction=true, tboundary=20, α1=1, α2=1, d=0u"bohr", showprogress=true)
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
        ct = optimal_coulomb_tranformation(ceg, nt; tmax=tmax, δ=δ, tboundary=tboundary)
    elseif  mode == :optimal
        @info "Optimal mode"
        ct = optimal_coulomb_tranformation(ceg, nt)
    else
        error("Mode not known")
    end
    return test_accuracy(ceg, ct; correction=correction, α1=α1, α2=α2, d=d, showprogress=showprogress)
end

function test_accuracy(ceg::CubicElementGrid, ct::AbstractCoulombTransformation;
                            α1=1, α2=1, d=0u"bohr", correction=true, showprogress=true)
    tmin = ct.tmin
    tmax = ct.tmax
    r = position_operator(ceg)
    r1 = r + [0.5, 0., 0.].*d
    r2 = r - [0.5, 0., 0.].*d
    q1 = QuantumState( exp( -0.5α1*u"bohr^-2"*(r1⋅r1) ) )
    q2 = QuantumState( exp( -0.5α1*u"bohr^-2"*(r2⋅r2) ) )
    sd = SlaterDeterminant(q1)
    V = coulomb_operator(sd, ct; correction=correction, showprogress=showprogress)
    V_corr = ScalarOperator(ceg, coulomb_correction(density_operator(q1).vals, ct.tmax); unit=u"hartree")

    E_cor = bracket(q2, V_corr, q2)
    E_int = bracket(q2, V, q2)
    E_tail = gaussian_coulomb_integral(α1, α2, austrip(d); tmin=tmax)[1] * u"hartree"
    E_true = gaussian_coulomb_integral(α1, α2, austrip(d); tmin=tmin, tmax=tmax)[1] * u"hartree"
    if correction
        E = E_int + E_cor
    else
        E = E_int
    end
    E_tot = gaussian_coulomb_integral(α1, α2, austrip(d))[1]*u"hartree"
    @info "Calculated energy = $E"
    @info "True inregration energy = $E_true"
    @info "Total Energy (reference) = $E_tot"
    @info "Integration error = $(round(u"hartree", (E_int-E_true); sigdigits=2)) ; error/E = $( round((E_int-E_true)/E_true; sigdigits=2))"
    @info "Integration error relative to total energy = $( round((E_int-E_true)/E_tot; sigdigits=2))"
    @info "Tail energy = $(round(u"hartree", E_tail; sigdigits=5)) ; E_tail/E_tot = $(round(E_tail/E_tot; sigdigits=2))"
    @info "Energy correction = $(round(u"hartree", E_cor; sigdigits=5))  ; (E_cor-E_tail)/E_tot = $(round((E_cor-E_tail)/E_tot;sigdigits=2))"
    @info "Error to total energy error/E_tot = $(round((E-E_tot)/E_tot; sigdigits=2))"
    out = Dict("calculated"=>E,
        "reference integral"=>E_true,
        "calculated integral"=>E_int,
        "correction"=>E_cor,
        "reference tail"=>E_tail,
        "total reference energy"=>E_tot
        )
    return out
end

function test_accuracy_ad(a, ne, ng, nt; tmax=300, α1=1, α2=1, d=0.5, δ=0.25, showprogress=true)
    ceg = CubicElementGrid(a, ne, ng)
    ct = optimal_coulomb_tranformation(ceg, nt; tmax=tmax, δ=δ, tboundary=20)
    return test_accuracy_ad(ceg::CubicElementGrid, ct::AbstractCoulombTransformation; α1=α1, α2=α2, d=d, showprogress=showprogress)
end


function test_accuracy_ad(ceg::CubicElementGrid, ct::AbstractCoulombTransformation;
                         α1=1., α2=1., d=0.5, showprogress=true)
    function f(x)
        tmin = ct.tmin
        tmax = ct.tmax
        r1 = SVector(0.5x[3], 0., 0.)
        r2 = SVector(-0.5x[3], 0., 0.)
        ρ1 = density_tensor(ceg; a=x[1], r=r1)
        ρ2 = density_tensor(ceg; a=x[2], r=r2)
        V = poisson_equation(ρ1, ct; tmax=tmax, showprogress=showprogress)
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


##  Kinetic energy tests
function test_kinetic_energy(a, ne, ng; ν=0, ω=1)
    #ceg = CubicElementGrid(a, ne, ng)
    ceg = ElementGridSymmetricBox(a, ne, ng)

    H = HamiltonOperatorFreeParticle(ceg)

    ψ = HarmonicEigenstate(ν; ω=ω)
    ϕ = ReferenceStates.harmonic_state(ceg, ψ)
    T = bracket(ϕ, H, ϕ)
    Tref = 0.5*3*ReferenceStates.energy(ψ)
    @info "Calculated kinetic energy = $T"
    @info "Reference kinetic energy = $Tref"
    @info "Error = $(T-Tref)"
    @info "Relative error = $((T-Tref)/Tref)"
    return T, Tref
end

# Wave function test
function test_atom_wavefunction(a, ne, nt; r=zeros(3), aname="H", ni=20)
    Z = elements[Symbol(aname)].number
    ceg = CubicElementGrid(a, ne, nt)
    ra = austrip.(r) .* u"bohr"
    r0 = position_operator(ceg) - ra
    rr = sqrt(r0⋅r0)

    V = -1u"e_au"*nuclear_potential(ceg, aname, r)

    phi = exp(-0.5u"bohr^-1"*rr)
    ϕ = QuantumState(ceg, phi.vals)
    normalize!(ϕ)

    H = HamiltonOperator(V)
    ψ = helmholtz_equation(ϕ, H; showprogress=true )

    for i in 1:ni
        helmholtz_equation!(ψ, H)
    end
    E = bracket(ψ, H, ψ) |> u"Ry"
    V = bracket(ψ, V, ψ) |> u"Ry"

    E_ref = Z^2 * u"Ry"

    @info "Energy = $E"
    @info "Reference energy = $E_ref"

    return Dict(
        "energy" => E,
        "E_ref"  => E_ref,
        "H"      => H,
        "ψ"      => ψ
    )
end
