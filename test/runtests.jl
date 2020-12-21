using Test
using CoulombIntegral
using Logging
using ForwardDiff

disable_logging(Logging.Info)


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

    # Symmetry
    @test ct.w[1]*ct[1,3,2,4,1] ≈ ct.w[3]*ct[3,1,4,2,1]
    @test ct.w[1]*clog[1,3,2,4,2] ≈ ct.w[3]*clog[3,1,4,2,2]
    @test ct.w[1]*ca[1,3,2,4,3] ≈ ct.w[3]*ca[3,1,4,2,3]
    @test ct.w[1]*cll[1,3,2,4,4] ≈ ct.w[3]*cll[3,1,4,2,4]
end

@testset "Operators" begin
    ceg = CubicElementGrid(10, 4, 16)
    r = position_operator(ceg)
    p = momentum_operator(ceg)
    x = r[1]
    @test x == r.operators[1]
    @test x != r.operators[2]
    @test x != r.operators[3]
    @test size(x) == size(r)
    @test_throws AssertionError r + p
    #@test sin(x).vals == sin.(x.vals)
    #for OP in [:sin, :cos, :exp]
    #    @eval begin
    #        v = $OP(x);
    #        vv = $OP.(x.vals);
    #    end
    #    @test v.vals == vv;
    #end
end


@testset "Poisson equation" begin
    ec = test_accuracy(10,4,24,48; showprogress=false)  # Poor accuracy
    eref = gaussian_coulomb_integral()[1]
    e = ec["calculated"]
    @test abs((e-eref)/eref) < 1e-3  # Calculation had poor accuracy
end

@testset "Forward mode AD" begin
    d = test_accuracy_ad(10, 4, 16, 96; α1=1.3, α2=0.8, d=0.5, showprogress=false)
    # Grid resolution is small so not very accurate
    @test all(abs.(d[1] .- d[2])./d[2] .< 1e-2)


    # Helmholtz equation derivative
    function f(k)
        ceg = CubicElementGrid(10,2,16)
        ct=loglocalct(ceg, 4; k=k)
        ρ = CoulombIntegral.density_tensor(ceg)
        V = poisson_equation(ρ, ct)
        return integrate(ρ, ceg, V)
    end
    g(x) = ForwardDiff.derivative(f, x)
    g(1.)
end

@testset "Derivative and kinetic energy" begin
    a = test_kinetic_energy(10, 4, 32; ν=1, ω=3)
    @test abs(a[1]-a[2]) < 1e-10
end
