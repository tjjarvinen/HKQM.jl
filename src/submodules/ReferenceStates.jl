module ReferenceStates

using Polynomials
using Unitful, UnitfulAtomic
using ..HKQM


export HarmonicEigenstate,
       harmonic_potential_well,
       harmonic_state

function hermite_polynomial(ν)
    if ν == 0
        return Polynomial([1])
    elseif ν == 1
        return Polynomial([0, 2])
    else
        p1 = Polynomial([0,2])*hermite_polynomial(ν-1)
        p2 = 2(ν-1)*hermite_polynomial(ν-2)
        return p1 - p2
    end
end


struct HarmonicEigenstate
    ν::Int
    α::Float64
    ω::Float64
    N::Float64
    hp::Polynomial{Int}
    function HarmonicEigenstate(ν::Int, ω=1)
        w = austrip(ω)
        α = 1/sqrt(w)
        hp = hermite_polynomial(ν)
        N = (2^ν*factorial(ν)*sqrt(π)*α)^(-1//2)
        new(ν, α, w, N, hp)
    end
end

function (HE::HarmonicEigenstate)(v::AbstractVector)
    return prod(HE, v)
end

function (HE::HarmonicEigenstate)(x::Real)
    y = x/HE.α
    return HE.N*HE.hp(y)*exp(-0.5*y^2)
end

function harmonic_state(ceg, hx, hy=hx, hz=hx)
    r = position_operator(ceg)
    ψ = map(r-> hx(r[1])*hy(r[2])*hz(r[3]), ceg)
    return QuantumState(ceg, ψ)
end

function energy(he::HarmonicEigenstate)
    return he.ω*(he.ν+0.5)*u"hartree"
end


"""
    harmonic_potential_well(ceg, ħω, x) -> ScalarOperator, Float64

Gives harmonic potential and estimate for lowest eigenvalue for associated Hamiltonian.
Potential is given so that all potential values are ≤ 0. Parameter `x` controls
how large area at the edges is put to zero = values over zero are cut.

# Arguments
- `ceg`    -   grid where potential is calculated
- `ħω`     -   ħω for the harmonic oscillator
- `x`      -   length of zero values at the edge

"""
function harmonic_potential_well(ceg, ħω, x)
    @assert dimension(x) == dimension(u"m")
    @assert dimension(ħω) == dimension(u"J")
    d = element_size( get_1d_grid(ceg) )
    x_max = 0.5 * d - x

    @assert 2x < d
    ω = ħω / 1u"ħ_au"

    # max n for bound state
    n = 1//2 * ( austrip( ω*x^2 ) -1 )
    @info "n < $( n )"

    k = ω^2 * 1u"me_au"
    E = 1//2 * k * x_max^2
    r = position_operator(ceg)
    r² = r⋅r
    V = 1//2  * k * r² - E
    for i in eachindex(V.vals)
        if V.vals[i] > 0
            V.vals[i] = 0
        end
    end
    E₀ = -E + 3//2 * ħω |> auconvert
    @info "E₀ = $E₀"
    return auconvert(V), E₀
end


end  # module
