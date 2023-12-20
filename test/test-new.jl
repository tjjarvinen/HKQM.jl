using Test
using Unitful
using HKQM.HelmholtzKernel



@testset "Elements" begin
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
            ele = Element1D(-2u"Å", 2u"Å")
            eg = @eval $(gt)($ele, 32)
            @test length(eg) == 32
            @test unit(eg) == unit(ele)

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

            #test integration
            w = get_weight(eg)
            c = cos.(eg)
            integral = sum( w .* c )
            a, b = ustrip.( element_bounds(eg) )
            @test integral ≈ sin(b) - sin(a)

            #test derivative
            D = get_derivative_matrix(eg)
            s = sin.(eg)
            c = D * s
            @test all( c .≈ cos.(eg) )
        end
    end
end


@testset "ElementGridVectors" begin
    gridtypes = [:ElementGridVectorLegendre, :ElementGridVectorLobatto]
    for gt in gridtypes
        @testset "$(gt)" begin
            ev = ElementVector(0.0u"pm", 1.5u"pm", 3.0u"pm")
            egv = @eval $(gt)($ev, 24)
            if gt == :ElementGridVectorLobatto
                # Lobatto basis combines points thus total is less
                @test length(egv) == 24*2 - 1
            else
                @test length(egv) == 24*2
            end
            @test eltype(egv) == Float64
            @test element_size(egv) ≈ element_size(ev)
            @test all( element_bounds(egv) .≈ element_bounds(ev) )
            @test unit(egv) == unit(ev)
            q = convert_variable_type(Float32, egv)
            @test eltype(q) == Float32
            D = get_derivative_matrix(q)
            @test size(D,1) == size(D,2) == length(egv)
            @test eltype(q) == Float32
            @test eltype(D) == Float32
            w = get_weight(q)
            @test eltype(w) == eltype(q)

            #test integration
            w = get_weight(egv)
            @test eltype(w) == eltype(egv)
            c = cos.(egv)
            integral = sum( w .* c )
            a, b = ustrip.( element_bounds(egv) )
            @test integral ≈ sin(b) - sin(a)

            #test derivative
            D = get_derivative_matrix(egv)
            @test eltype(D) == eltype(egv)
            s = sin.(egv)
            c = D * s
            @test all( c .≈ cos.(egv) )
        end
    end
end

@testset "ElementGridArray" begin
    ev = ElementVector(0.0u"pm", 1.5u"pm", 3.0u"pm")
    egv = ElementGridVectorLegendre(ev, 24)
    ego = ElementGridVectorLobatto(ev, 24)
    ega = ElementGridArray(egv, ego, egv)
    @test size(ega) == (length(egv), length(ego), length(egv))
    @test all( ega[1,3,5] .== [egv[1], ego[3], egv[5]] )
    @test get_weight(ega,1) ≈ get_weight(egv)
    @test all( element_bounds(ega,1) .≈ element_bounds(egv) )
    @test get_derivative_matrix(ega,2) ≈ get_derivative_matrix(ego)
    @test get_derivative_matrix(ega,1) ≈ get_derivative_matrix(egv)

    d = fill(ega, 1.0)
    @test integrate(ega, d) ≈ 27.0
end
