using Test
using CoulombIntegral
using ForwardDiff
using TensorOperations


@testset "Elements" begin
    ce = CubicElements(10, 4)
    @test size(ce) == (4,4,4)

    ceg = CubicElementGrid(10, 4, 16)
    @test size(ceg) == (16,16,16,4,4,4)

end

@testset "Tensors" begin
    ceg = CubicElementGrid(10, 4, 16)
    ct = CoulombTransformation(ceg, 16)
    @test size(ct) == (16,16,4,4,16)
end

@testset "Forward mode AD" begin
    function testad(a)
        tmax=25
        ceg = CubicElementGrid(10, 4, 16)
        ct = CoulombTransformation(ceg, 16; tmax=tmax)
        ρ = CoulombIntegral.density_tensor(Array(ceg), zeros(3), a[1])
        V = coulomb_tensor(ρ, Array(ct), Array(ct.t))
        V = V .+ coulomb_correction(ρ, tmax)

        ω = ω_tensor(ceg)

        cc = V.*ρ
        return @tensor ω[α,I]*ω[β,J]*ω[γ,K]*cc[α,β,γ,I,J,K]
    end

    g = x -> ForwardDiff.gradient(testad, [x]);

    @test g(1.) != 0.
end
