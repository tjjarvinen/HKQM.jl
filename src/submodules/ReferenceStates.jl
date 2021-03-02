module ReferenceStates

using Polynomials
using Unitful, UnitfulAtomic


export HarmonicEigenstate,
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
    function HarmonicEigenstate(ν::Int; ω=1)
        α = 1/sqrt(ω)
        hp = hermite_polynomial(ν)
        N = (2^ν*factorial(ν)*sqrt(π)*α)^(-1//2)
        new(ν, α, ω, N, hp)
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
    ψ = map(r-> hx(r[1])*hy(r[2])*hz[3], ceg)
    return ψ
end

function energy(he::HarmonicEigenstate)
    return he.ω*(he.ν+0.5)*u"hartree"
end


end  # module
