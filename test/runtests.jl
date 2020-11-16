using Test
using CoulombIntegral


@testset "Elements" begin
    ce = CubicElements(10, 4)
    @test size(ce) == (4,4,4)

    ceg = CubicElementGrid(10, 4, 16)
    @test size(ceg) == (16,16,16,4,4,4)

end

@testset "Tensors" begin
    ceg = CubicElementGrid(10, 4, 16)
    ct = CoulombTransformation(ceg, 16)
    clog = CoulombTransformationLog(ceg, 16)
    ca = CoulombTransformationLocal(ceg, 16)
    cll  = CoulombTransformationLogLocal(ceg, 16)
    @test size(ct) == (16,16,4,4,16)
end

@testset "Poisson equation" begin
    e = test_accuracy(10,4,24,48)  # Poor accuracy
    eref = gaussian_coulomb_integral()[1]
    @test abs((e-eref)/eref) < 1e-3  # Calculation had poor accuracy
end

@testset "Forward mode AD" begin
    d = test_accuracy_ad(10, 4, 16, 96; α1=1.3, α2=0.8, d=0.5)
    # Grid resolution is small so not very accurate
    @test all(abs.(d[1] .- d[2])./d[2] .< 1e-2)
end

@testset "Derivative and kinetic energy" begin
    a = test_kinetic_energy(10, 4, 32; ν=1, ω=3)
    @test abs(a[1]-a[2]) < 1e-10
end
