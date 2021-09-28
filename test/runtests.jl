using Test
using HKQM
using Logging
using ForwardDiff


disable_logging(Logging.Info)


@testset "Elements" begin
    ca = CubicElementArray(5u"Å", 3)
    @test size(ca) == (3,3,3)
    @test get_center(ca[2,2,2]) ≈ zeros(3).*u"bohr"

    ceg = CubicElementGrid(5u"Å", 4, 16)
    @test size(ceg) == (16,16,16,4,4,4)

    ev = ElementVector(0, 2, 5, 7)
    @test length(ev) == 3

    egv = ElementGridVector(ev, 6)
    @test size(egv) == (6,3)
    @test egv[begin, begin] >= 0
    @test egv[end, end] <= 7
end

@testset "Tensors" begin
    ceg = CubicElementGrid(5u"Å", 4, 16)
    ct = CoulombTransformation(ceg, 16)
    clog = CoulombTransformationLog(ceg, 16)
    ca = CoulombTransformationLocal(ceg, 16)
    cll  = CoulombTransformationLogLocal(ceg, 16)
    @test size(ct) == (16,16,4,4,16)
    @test size(ct) == size(clog)
    @test size(ct) == size(ca)
    @test size(ct) == size(cll)

    # Symmetry
    @test ct.w[1]*ct[1,3,2,4,1] ≈ ct.w[3]*ct[3,1,4,2,1]
    @test ct.w[1]*clog[1,3,2,4,2] ≈ ct.w[3]*clog[3,1,4,2,2]
    @test ct.w[1]*ca[1,3,2,4,3] ≈ ct.w[3]*ca[3,1,4,2,3]
    @test ct.w[1]*cll[1,3,2,4,4] ≈ ct.w[3]*cll[3,1,4,2,4]
end

@testset "Operators" begin
    ceg = CubicElementGrid(5u"Å", 4, 16)
    r = position_operator(ceg)
    p = momentum_operator(ceg)
    x = r[1]
    @test unit(x) == u"bohr"
    @test x == r.operators[1]
    @test x != r.operators[2]
    @test x != r.operators[3]
    @test size(x) == size(r)
    @test_throws AssertionError r + p
    @test sin(1u"bohr^-1"*x).vals == sin.(x.vals)
    @test cos(1u"bohr^-1"*x).vals == cos.(x.vals)
    @test exp(1u"bohr^-1"*x).vals == exp.(x.vals)
    @test (x + x).vals == 2.0 .* x.vals
    @test (2x).vals ==  (x + x).vals
    @test (x^2).vals == x.vals.^2
    @test sqrt(x^2).vals ≈ abs.(x.vals)
    @test (x^2).vals == (x*x).vals
    @test unit(x^2) == u"bohr^2"
    @test unit(x+x) == u"bohr"
    @test unit(sqrt(x^2)) == u"bohr"
end

@testset "Quantum States" begin
    ceg = CubicElementGrid(5u"Å", 4, 16)
    r = position_operator(ceg)
    r² = r⋅r
    gv = exp(-1u"bohr^-2"*r²)
    ψ = QuantumState(ceg, gv.vals)
    normalize!(ψ)
    @test bracket(ψ,ψ) ≈ 1
    ϕ = ψ + 2ψ
    @test bracket(ϕ,ψ) ≈ 3
    rr = bracket(ψ, r, ψ)
    @test sqrt( sum(x->x^2, rr) ) < 1u"bohr"*1E-12
end

@testset "Nuclear potential" begin
    tn = test_nuclear_potential(5u"Å", 4, 64, 64; mode="preset")
    @test abs( (tn["integral"] - tn["total reference"]) / tn["total reference"]) < 1e-6
end


@testset "Poisson equation" begin
    ec = test_accuracy(5u"Å",4,24,48; showprogress=false)  # Poor accuracy
    eref = gaussian_coulomb_integral()[1]
    e = ec["calculated"]
    @test abs((e-eref)/eref) < 1e-3  # Calculation had poor accuracy

    nc = test_nuclear_potential(5u"Å", 2 ,64, 64; mode="preset")
    @test abs(nc["integral"] - nc["total reference"])/nc["total reference"] < 1e-5
end

@testset "Forward mode AD" begin
    d = test_accuracy_ad(5u"Å", 4, 16, 96; α1=1.3, α2=0.8, d=0.5, showprogress=false)
    # Grid resolution is small so not very accurate
    @test all(abs.(d[1] .- d[2])./d[2] .< 1e-2)


    # Helmholtz equation derivative
    function f(k)
        ceg = CubicElementGrid(5u"Å",2,16)
        ct=optimal_coulomb_tranformation(ceg, 4; k=k)
        ρ = HKQM.density_tensor(ceg)
        V = poisson_equation(ρ, ct)
        return HKQM.integrate(ρ, ceg, V)
    end
    g(x) = ForwardDiff.derivative(f, x)
    g(1.)
end

@testset "Derivative and kinetic energy" begin
    a = test_kinetic_energy(5u"Å", 4, 32; ν=1, ω=3)
    @test abs(a[1]-a[2]) < 1e-10
end
