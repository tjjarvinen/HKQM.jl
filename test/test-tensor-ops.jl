using TensorOperations


@testset "Quantum States - TensorOperations" begin
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

@testset "Float32 support - TensorOperations" begin
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

@testset "Derivatives - TensorOperations" begin
    ceg = ElementGridSymmetricBox(5u"Å", 4, 24)
    r = position_operator(ceg)
    lo = LaplaceOperator(ceg)

    # x derivatives
    x = r[1] * 1u"bohr^-1"
    s = QuantumState(sin(x))
    dx = DerivativeOperator(ceg,1)
    @test (cos(x)).vals ≈ (dx*s).psi # cos(x) = D*sin(x) 
    @test s.psi ≈ (-lo*s).psi  # sin(x) = -∇²sin(x)

    # y derivatives
    y = r[2] * 1u"bohr^-1"
    s = QuantumState(sin(y))
    dy = DerivativeOperator(ceg,2)
    @test (cos(y)).vals ≈ (dy*s).psi
    @test s.psi ≈ (-lo*s).psi

    # z derivatives
    z = r[3] * 1u"bohr^-1"
    s = QuantumState(sin(z))
    dz = DerivativeOperator(ceg,3)
    @test (cos(z)).vals ≈ (dz*s).psi
    @test s.psi ≈ (-lo*s).psi
end


@testset "Poisson equation - TensorOperations" begin
    ec = test_accuracy(5u"Å",4,24,48; showprogress=false)  # Poor accuracy
    eref = gaussian_coulomb_integral()[1]*u"hartree"
    e = ec["calculated"]
    @test abs((e-eref)/eref) < 1e-3  # Calculation had poor accuracy
end

@testset "Forward mode AD - TensorOperations" begin
    d = test_accuracy_ad(5u"Å", 4, 16, 96; α1=1.3, α2=0.8, d=0.5, showprogress=false)
    # Grid resolution is small so not very accurate
    @test all(abs.(d[1] .- d[2])./d[2] .< 1e-2)


    # Helmholtz equation derivative
    function f(k)
        ceg = ElementGridSymmetricBox(5u"Å",2,16)
        ct = optimal_coulomb_tranformation(ceg, 4; k=k)
        ρ = particle_in_box(ceg, 1,2,3)
        V = poisson_equation(ρ, ct)
        return bracket(ρ, V) |> ustrip
    end
    g(x) = ForwardDiff.derivative(f, x)
    @test g(1.) < 0
end

@testset "Derivative and kinetic energy - TensorOperations" begin
    gridtypes = [:ElementGridSymmetricBox]
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


@testset "1D Schrödinger equation - TensorOperations" begin
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


@testset "Hartree-Fock - TensorOperations" begin
    ceg = ElementGridSymmetricBox(5u"Å", 2, 32)

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