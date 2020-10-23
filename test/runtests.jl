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
    @test size(ct) == (16,16,4,4,16)
end

@testset "Forward mode AD" begin
    d = test_accuracy_ad(10, 4, 16, 16; mode=:log, ae=1.3)
    @test abs(d[1] - d[2]) < 1e-3
end
