@testset "Elements" begin
    @testset "Element1D" begin
        e = Element1D(-2u"m", 3.1u"m")
        q = uconvert(u"Å", e)
        @test element_size(e) ≈ 3.1u"m" - (-2u"m")
        @test element_size(q) ≈ element_size(e)
        @test unit(e) == u"m"
        @test unit(q) == u"Å"
        @test all( element_bounds(e) .≈ (-2u"m", 3.1u"m") )
    end
    @testset "ElementVector" begin
        e = Element1D(-2u"m", 3.1u"m")
        q = Element1D(31u"dm", 50u"dm")
        ev = ElementVector(e, q)
        @test length(ev) == 2
        @test element_size(ev) ≈ element_size(e) + element_size(q)
        @test element_size(ev) ≈ sum(x->element_size(x), ev)
        @test unit(ev) == unit(e)
        @test unit(ev[1]) == unit(e)
        @test unit(ev[2]) == unit(e)
        @test unit( uconvert(u"Å", ev) ) == u"Å"
        @test all( element_bounds(ev) .≈ (-2u"m", 5u"m") )

        evv = ElementVector(0u"m", 2u"m", 5u"m", 7u"m")
        @test length(evv) == 3
    end
    @testset "ElementGrids" begin
        gridtypes = [:ElementGridLegendre, :ElementGridLobatto]
        for gt in gridtypes
            @testset "$(gt)" begin
                eg = @eval $(gt)(Element1D(-2u"Å", 2u"Å"), 32)
                @test length(eg) == 32
                @test unit(eg) == u"bohr"
                ee = @eval $(gt)(Element1D(-2u"Å", 2u"Å"), 32; unit=u"Å")
                @test unit(ee) == u"Å"

                @test maximum(ee) * unit(ee) <=  2u"Å"
                @test minimum(ee) * unit(ee) >= -2u"Å" 
                @test maximum(eg) * unit(eg) <= 2u"Å"
                @test minimum(eg) * unit(eg) >= -2u"Å"

                @test all( element_bounds(eg) .≈ (-2u"Å", 2u"Å") )
                @test element_size(eg) ≈ 4u"Å"
                q = convert_variable_type(Float32, eg)
                @test eltype(q) == Float32
                @test eltype(eg) == Float64

                #test interpolation
                u = sin.(eg)
                @test eg(1.1, u) ≈ sin(1.1)
            end
        end
    end

    ca = CubicElementArray(5u"Å", 3)
    @test size(ca) == (3,3,3)
    @test get_center(ca[2,2,2]) ≈ zeros(3).*u"bohr"

    ceg = CubicElementGrid(5u"Å", 4, 16)
    @test size(ceg) == (16,16,16,4,4,4)

    ev = ElementVector(0, 2, 5, 7)

    eg = ElementGridLegendre(0, 3, 32)
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
    @test sqrt( sum(x->x^2, rr) ) < 1u"bohr"*1E-7

    ceg_f64 = convert_variable_type(Float64, ceg)
    @test ceg_f64 ≈ ceg

    ψ_f64 = convert_variable_type(Float64, ψ)
    @test ψ_f64.psi ≈ ψ.psi
    @test unit(ψ_f64) == unit(ψ)
    @test get_elementgrid(ψ_f64) ≈ get_elementgrid(ψ)
end

@testset "Derivatives" begin
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
        ceg = ElementGridSymmetricBox(5u"Å",2,16)
        ct = optimal_coulomb_tranformation(ceg, 4; k=k)
        ρ = particle_in_box(ceg, 1,2,3)
        V = poisson_equation(ρ, ct)
        return bracket(ρ, V) |> ustrip
    end
    g(x) = ForwardDiff.derivative(f, x)
    @test g(1.) < 0  skip=true
end

@testset "Derivative and kinetic energy" begin
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


@testset "Reference Potential" begin
    ceg = ElementGridSymmetricBox(5u"Å", 4, 24)

    V, E = HKQM.ReferenceStates.harmonic_potential_well(ceg, 100u"eV", 1u"Å")

    @test V.vals[1,1,1,1,1,1] ≈ 0
end