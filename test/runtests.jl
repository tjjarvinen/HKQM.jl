using CoulombIntegral
using Test

@testset "Elements" begin
    ce = CubicElements(-5,5,4)
    @test size(ce) == (4,4,4)
    @test ce[1,2,3][3] == ce[1,4,3][3]
    @test ce[2,1,3] != ce[1,2,3]

    eg = CubicElementGrid(-10,10, 4, 16)
    @test size(eq) == (16,16,16,4,4,4)

end
