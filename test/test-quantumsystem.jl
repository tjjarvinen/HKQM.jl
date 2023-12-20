using Test
using HKQM.HelmholtzKernel
using HKQM.QuantumSystem
using Unitful


@testset "Quantum States" begin
    ev = ElementVector(0.0u"pm", 1.5u"pm", 3.0u"pm")
    ego = ElementGridVectorLobatto(ev, 24)
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