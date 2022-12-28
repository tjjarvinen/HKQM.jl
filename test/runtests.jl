using Test
using HKQM
using Logging
using ForwardDiff
using SpecialFunctions


disable_logging(Logging.Info)




@testset "Elements" begin
    ca = CubicElementArray(5u"Å", 3)
    @test size(ca) == (3,3,3)
    @test get_center(ca[2,2,2]) ≈ zeros(3).*u"bohr"

    ceg = CubicElementGrid(5u"Å", 4, 16)
    @test size(ceg) == (16,16,16,4,4,4)

    ev = ElementVector(0, 2, 5, 7)
    @test length(ev) == 3

    eg = ElementGrid(0, 3, 32)
    u = sin.(eg)
    x = range(0,3; length=200)
    @test sin.(x) ≈ eg(x, u)

    egv = ElementGridVector(ev, 6)
    @test size(egv) == (6,3)
    @test egv[begin, begin] >= 0
    @test egv[end, end] <= 7
end

@testset "Tensors" begin
    gridtypes = [:ElementGridSymmetricBox, :CubicElementGrid]
    for gt in gridtypes
        @testset "$(gt)" begin
            ceg = @eval $(gt)(5u"Å", 4, 16)
            ct = HelmholtzTensorLinear(ceg, 16)
            clog = HelmholtzTensorLog(ceg, 16)
            ca = HelmholtzTensorLocalLinear(ceg, 16)
            cll  = HelmholtzTensorLocalLog(ceg, 16)
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
    end
end

@testset "Operators" begin
    gridtypes = [:ElementGridSymmetricBox, :CubicElementGrid]
    for gt in gridtypes
        @testset "$(gt)" begin
            ceg = @eval $(gt)(5u"Å", 4, 16)
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
    end
end

@testset "Quantum States" begin
    ceg = ElementGridSymmetricBox(5u"Å", 4, 16)
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

@testset "Float32 support" begin
    ceg = ElementGridSymmetricBox(Float32, 5u"Å", 4, 16)
    r = position_operator(ceg)
    @test eltype(r[1].vals) == Float32
    r² = r⋅r
    gv = exp(-1u"bohr^-2"*r²)
    ψ = QuantumState(ceg, gv.vals)
    @test eltype(ψ.psi) == Float32
    normalize!(ψ)
    @test bracket(ψ,ψ) ≈ 1
    ϕ = ψ + 2ψ
    @test eltype(ϕ.psi) == Float32
    @test bracket(ϕ,ψ) ≈ 3
    rr = bracket(ψ, r, ψ)
    @test typeof(austrip(rr[1])) == Float32
    @test sqrt( sum(x->x^2, rr) ) < 1u"bohr"*1E-8

    ceg_f64 = convert_variable_type(Float64, ceg)
    @test ceg_f64 ≈ ceg

    ψ_f64 = convert_variable_type(Float64, ψ)
    @test ψ_f64.psi ≈ ψ.psi
    @test unit(ψ_f64) == unit(ψ)
    @test get_elementgrid(ψ_f64) ≈ get_elementgrid(ψ)
end

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


@testset "Poisson equation" begin
    ec = test_accuracy(5u"Å",4,24,48; showprogress=false)  # Poor accuracy
    eref = gaussian_coulomb_integral()[1]*u"hartree"
    e = ec["calculated"]
    @test abs((e-eref)/eref) < 1e-3  # Calculation had poor accuracy
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
    gridtypes = [:CubicElementGrid, :ElementGridSymmetricBox]
    for gt in gridtypes
        ceg = @eval $(gt)(5u"Å", 4, 24)
        H = HamiltonOperatorFreeParticle(ceg)
        ψ = HKQM.ReferenceStates.HarmonicEigenstate(1, 3)
        ϕ = HKQM.ReferenceStates.harmonic_state(ceg, ψ)
        T = bracket(ϕ, H, ϕ)
        Tref = 0.5*3*HKQM.ReferenceStates.energy(ψ)
        @test abs( (T-Tref) / Tref ) < 1e-10
    end
end


@testset "1D Schrödinger equation" begin
    ceg = ElementGridSymmetricBox(5u"Å", 4, 24)
    r = position_operator(ceg)    
    r² = r ⋅ r 

    V = -30u"eV" * exp(-0.1u"bohr^-2" * r²) 
    H = HamiltonOperator(V)

    W111 = particle_in_box(ceg, 1,1,1)
    W112 = particle_in_box(ceg, 1,1,2)

    estates, evals = solve_eigen_states(H, W111, W112; max_iter=2)
    @test bracket(W111, H, W111) > evals[1]
    @test bracket(W112, H, W112) > evals[2]
end


@testset "Hartree-Fock" begin
    ceg = CubicElementGrid(5u"Å", 2, 32)

    r = position_operator(ceg)

    V = -10u"hartree"*exp(-1u"Å^-2"*r⋅r)

    n0 = HKQM.ReferenceStates.HarmonicEigenstate(0)
    n1 = HKQM.ReferenceStates.HarmonicEigenstate(1)
    q0 = HKQM.ReferenceStates.harmonic_state(ceg, n0, n0, n0)
    q1 = HKQM.ReferenceStates.harmonic_state(ceg, n0, n1, n0)

    H = HamiltonOperator(V)
    sd = SlaterDeterminant(q0,q1)
    sd1 = scf(sd, H; max_iter=2)
    S = overlap_matrix(sd1)
    @test bracket(H,sd) >  bracket(H, sd1)
end


@testset "Reference Potential" begin
    ceg = ElementGridSymmetricBox(5u"Å", 4, 24)

    V, E = HKQM.ReferenceStates.harmonic_potential_well(ceg, 100u"eV", 1u"Å")

    @test V.vals[1,1,1,1,1,1] ≈ 0
end