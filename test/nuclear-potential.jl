using Tullio

@testset "Nuclear potential" begin
    ceg = ElementGridSymmetricBox(5u"Å", 4, 24)
    V = nuclear_potential_harrison_approximation(ceg, zeros(3)*u"bohr", "C")
    
    # Ref value
    rt = ceg[1,2,3,3,3,3]
    c  = cbrt( 0.00435 * 1E-6 / 6^5 )
    r² = sum(x->x^2, rt) / c^2
    r  = sqrt(r²)
    ref = -6/c * erf(r)/r + 1/(3*√π) * ( exp(-r²) + 16exp(-4r²) )

    @test V.vals[1,2,3,3,3,3] ≈ ref

    tn = test_nuclear_potential(5u"Å", 4, 64, 64; mode="preset")
    @test abs( (tn["integral"] - tn["total reference"]) / tn["total reference"]) < 1e-6
end