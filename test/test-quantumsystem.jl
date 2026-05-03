using Test
using HKQM.HelmholtzKernel
using HKQM.QuantumSystem
using LinearAlgebra
using Unitful


@testset "Quantum States" begin
    ev = ElementVector(0.0u"pm", 1.5u"pm", 3.0u"pm")
    ego = ElementGridVectorLobatto(ev, 16)
    ega = ElementGridArray(ego, ego, ego)

    psi = QuantumState(ega, fill(ega,1.0im), u"pm")

    @test braket(psi, psi) ≈ 27.0 * unit(psi)^2
    cpsi = conj(psi)
    @test unit(cpsi) == unit(psi)
    @test cpsi[1] == -im
    conj!(cpsi)
    @test cpsi[1] == im

    psi2 = 2*psi
    @test unit(psi2) == unit(psi)
    @test psi2[1] ≈ 2 * psi[1]
    psi1 = psi2 / 2
    @test psi1[1] ≈ psi[1]
    q = psi + psi
    @test unit(q) == unit(psi)
    @test q[1] == psi2[1]

    q = psi - psi
    @test unit(q) == unit(psi)
    @test braket(q, q) ≈ 0.0 * unit(q)^2

    normalize!(psi)
    @test braket(psi, psi) ≈ 1.0
end


@testset "Operators" begin
    ev = ElementVector(0.0u"pm", 1.5u"pm", 3.0u"pm")
    ego = ElementGridVectorLobatto(ev, 16)
    ega = ElementGridArray(ego, ego, ego)

    r = position_operator(ega)
    p = momentum_operator(ega)

    psi = particle_in_box(ega, 1, 2, 3)

    psi1 = r[1] * psi
    psi2 = p[1] * psi

    @test braket(psi, r[1], psi) ≈ 1.5u"pm"
    @test braket(psi, r[2], psi) ≈ 1.5u"pm"
    @test braket(psi, r[3], psi) ≈ 1.5u"pm"

    r2 = dot(r, r)

    @test dimension(r2) == dimension(r)^2

    @test braket(psi, -r2, psi) < 0.0u"pm^2"

    lo = LaplaceOperator()
    @test braket(psi, lo, psi) < 0.0u"pm^-2"

    grad = gradient_operator(ega)
    @test length( braket(psi, grad, psi) ) == 3
end