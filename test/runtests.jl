using CoulombIntegral
using Test

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
