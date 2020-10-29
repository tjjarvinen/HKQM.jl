using TensorOperations
using ProgressMeter
using OffsetArrays
using SpecialFunctions
using StaticArrays


abstract type AbtractTransformationTensor{T} <: AbstractArray{Float64, T} end

abstract type AbstractCoulombTransformation <: AbtractTransformationTensor{5} end

abstract type AbstractCoulombTransformationSingle{NT, NE, NG}  <: AbstractCoulombTransformation where {NT, NE, NG} end
abstract type AbstractCoulombTransformationCombination <: AbstractCoulombTransformation end
abstract type AbstractCoulombTransformationLocal{NT, NE, NG} <: AbstractCoulombTransformationSingle{NT, NE, NG} end

struct CoulombTransformation{NT, NE, NG} <: AbstractCoulombTransformationSingle{NT, NE, NG}
    elementgrid::SMatrix{NG,NE}
    t::SVector{NT}{Float64}
    wt::SVector{NT}{Float64}
    w::SVector{NG}{Float64}
    tmin::Float64
    tmax::Float64
    function CoulombTransformation(ceg::CubicElementGrid, nt::Int; tmin=0., tmax=25.)
        t, wt = gausspoints(nt; elementsize=(tmin, tmax))
        grid = grid1d(ceg)
        s = size(grid)
        new{nt, s[2], s[1]}(grid, t, wt, ceg.w, tmin, tmax)
    end
end

struct CoulombTransformationLog{NT, NE, NG} <: AbstractCoulombTransformationSingle{NT, NE, NG}
    elementgrid::SMatrix{NG,NE}
    t::SVector{NT}{Float64}
    wt::SVector{NT}{Float64}
    w::SVector{NG}{Float64}
    tmin::Float64
    tmax::Float64
    function CoulombTransformationLog(ceg::CubicElementGrid, nt::Int;
                                   tmin=0., tmax=25., ε=1e-12)
        s, ws = gausspoints(nt; elementsize=(log(tmin+ε), log(tmax)))
        t = exp.(s)
        wt = ws .* t
        grid = grid1d(ceg)
        ss = size(grid)
        new{nt, ss[2], ss[1]}(grid, t, wt, ceg.w, tmin, tmax)
    end
end

struct CoulombTransformationLocal{NT, NE, NG} <: AbstractCoulombTransformationLocal{NT, NE, NG}
    elementgrid::SMatrix{NG,NE}
    t::SVector{NT}{Float64}
    wt::SVector{NT}{Float64}
    w::SVector{NG}{Float64}
    tmin::Float64
    tmax::Float64
    δ::Float64
    δp::SMatrix{NG,NE}
    δm::SMatrix{NG,NE}
    function CoulombTransformationLocal(ceg::CubicElementGrid, nt::Int;
                                   tmin=0., tmax=25., δ=0.25)
        @assert 0 < δ <= 1
        @assert 0 <= tmin < tmax
        t, wt = gausspoints(nt; elementsize=(tmin, tmax))
        grid = grid1d(ceg)
        s = size(grid)
        δp = zeros(s...)
        δm = zeros(s...)
        δm[2:end,:] = grid[1:end-1,:] .- grid[2:end,:]
        δp[1:end-1,:] = grid[2:end,:] .- grid[1:end-1,:]
        for j in 1:s[2]
            δm[1,j] = ceg.elements[j].low - grid[1,j]
            δp[end,j] = ceg.elements[j].high - grid[end,j]
        end
        new{nt, s[2], s[1]}(grid, t, wt, ceg.w, tmin, tmax, δ, δ.*δp, δ.*δm)
    end
end

struct CoulombTransformationLogLocal{NT, NE, NG} <: AbstractCoulombTransformationLocal{NT, NE, NG}
    elementgrid::SMatrix{NG,NE}
    t::SVector{NT}{Float64}
    wt::SVector{NT}{Float64}
    w::SVector{NG}{Float64}
    tmin::Float64
    tmax::Float64
    δ::Float64
    δp::SMatrix{NG,NE}
    δm::SMatrix{NG,NE}
    function CoulombTransformationLogLocal(ceg::CubicElementGrid, nt::Int;
                                   tmin=0., tmax=25., δ=0.25, ε=1e-12)
        @assert 0 < δ <= 1
        @assert 0 <= tmin < tmax
        s, ws = gausspoints(nt; elementsize=(log(tmin+ε), log(tmax)))
        t = exp.(s)
        wt = ws .* t
        t, wt = gausspoints(nt; elementsize=(tmin, tmax))
        grid = grid1d(ceg)
        ss = size(grid)
        δp = zeros(ss...)
        δm = zeros(ss...)
        δm[2:end,:] = grid[1:end-1,:] .- grid[2:end,:]
        δp[1:end-1,:] = grid[2:end,:] .- grid[1:end-1,:]
        for j in 1:ss[2]
            δm[1,j] = ceg.elements[j].low - grid[1,j]
            δp[end,j] = ceg.elements[j].high - grid[end,j]
        end
        new{nt, ss[2], ss[1]}(grid, t, wt, ceg.w, tmin, tmax, δ, δ.*δp, δ.*δm)
    end
end

mutable struct CoulombTransformationCombination{NE, NG} <: AbstractCoulombTransformationCombination
    tensors::Vector{AbstractCoulombTransformation}
    ref::Vector{Int}
    t::Vector{Float64}
    wt::Vector{Float64}
    tmin::Float64
    tmax::Float64
    tindex::Vector{Int}
    function CoulombTransformationCombination(ct::AbstractCoulombTransformationSingle)
        s = size(ct)
        new{s[3], s[2]}([ct], ones(s[end]), ct.t, ct.wt, ct.tmin, ct.tmax, 1:length(ct.t))
    end
end

function CoulombTransformationCombination(t...)
    ct = CoulombTransformationCombination(t[1])
    for x ∈ t[2:end]
        push!(ct,x)
    end
    return ct
end

function loglocalct(ceg::CubicElementGrid, nt::Int; δ=0.25, tmax=300, tboundary=20, ε=1e-12)
    @assert 0 < tboundary < tmax
    s, ws = gausspoints(nt; elementsize=(log(ε), log(tmax)))
    t = exp.(s)
    l = length(t[t .< tboundary])
    ct1 = CoulombTransformationLog(ceg, l, tmax=tboundary, ε=ε)
    ct2 = CoulombTransformationLogLocal(ceg, nt-l; tmin=tboundary, tmax=tmax, δ=δ)
    return CoulombTransformationCombination(ct1,ct2)
end



(ct::AbstractCoulombTransformation)(r,t) = exp(-(t*r)^2)
(ct::AbstractCoulombTransformationSingle)(r,p::Int) = exp(-ct.t[p]^2*r^2)

function (ct::AbstractCoulombTransformationLocal)(r,t)
    rmax = (r+ct.δ)*t
    rmin = (r-ct.δ)*t
    return 0.5*√π*erf(rmin, rmax)/(rmax-rmin)
end

function (ct::AbstractCoulombTransformationLocal)(r,p::Int)
    rmax = (r+ct.δ)*ct.t[p]
    rmin = (r-ct.δ)*ct.t[p]
    return 0.5*√π*erf(rmin, rmax)/(rmax-rmin)
end

function (ct::CoulombTransformationCombination)(r,p::Int)
    i = ct.ref[p]
    return ct.tensors[i](r, ct.t[p])
end

function Base.size(ct::AbstractCoulombTransformationSingle)
    ng, ne = size(ct.elementgrid)
    nt = length(ct.t)
    return (ng,ng,ne,ne,nt)
end

function Base.size(ct::CoulombTransformationCombination{NE,NG}) where {NE,NG}
    nt = length(ct.ref)
    return (NG,NG,NE,NE,nt)
end

function Base.getindex(ct::AbstractCoulombTransformationSingle, α::Int, β::Int, I::Int, J::Int, p::Int)
    r = ct.elementgrid[α,I] - ct.elementgrid[β,J]
    return ct.w[β]*ct(r, p)
end

function Base.getindex(ct::AbstractCoulombTransformationLocal, α::Int, β::Int, I::Int, J::Int, p::Int)
    r = abs(ct.elementgrid[α,I] - ct.elementgrid[β,J])
    δ = min(abs(ct.δp[α,I]-ct.δm[β,J]), abs(ct.δm[α,I]-ct.δp[β,J]))
    rmax = (r+δ)*ct.t[p]
    rmin = (r-δ)*ct.t[p]
    return ct.w[β]*0.5*√π*erf(rmin, rmax)/(rmax-rmin)
end

function Base.getindex(ct::CoulombTransformationCombination, α::Int, β::Int, I::Int, J::Int, p::Int)
    i = ct.ref[p]
    return ct.tensors[i][α, β, I, J, ct.tindex[p]]
end

function  Base.push!(ctc::CoulombTransformationCombination{NE,NG},
            tt::AbstractCoulombTransformationSingle{NT,NE,NG}) where {NT,NE,NG}
    @assert ctc.tensors[end].tmax <= tt.tmin
    push!(ctc.tensors,tt)
    ctc.ref = vcat(ctc.ref, ones(NT).*length(ctc.tensors))
    ctc.t = vcat( ctc.t, tt.t)
    ctc.wt = vcat( ctc.wt, tt.wt)
    ctc.tmax = tt.tmax
    ctc.tindex = vcat( ctc.tindex, 1:length(tt.t))
    return ctc
end


function Base.show(io::IO, ::MIME"text/plain", cct::CoulombTransformation)
    print(io, "Coulomb transformation tensor size=$(size(cct))")
end

function Base.show(io::IO, ::MIME"text/plain", cct::CoulombTransformationLog)
    print(io, "Coulomb transformation tensor logaritmic size=$(size(cct))")
end

function Base.show(io::IO, ::MIME"text/plain", cct::CoulombTransformationLocal)
    print(io, "Coulomb transformation tensor local means size=$(size(cct))")
end

function Base.show(io::IO, ::MIME"text/plain", cct::CoulombTransformationCombination)
    print(io, "Coulomb transformation combination tensor size=$(size(cct))")
end

function Base.show(io::IO, ::MIME"text/plain", cct::CoulombTransformationLogLocal)
    print(io, "Coulomb transformation logarimic local size=$(size(cct))")
end


function ω_tensor(ceg::CubicElementGrid)
    n = size(ceg)[end]
    l = length(ceg.w)
    ω = reshape( repeat(ceg.w, n), l, n )
end


function integrate(ϕ, grid::CubicElementGrid, ψ)
    ω = ω_tensor(grid)
    c = ϕ.*ψ
    return  @tensor ω[α,I]*ω[β,J]*ω[γ,K]*c[α,β,γ,I,J,K]
end



function density_tensor(grid::AbstractArray, r::AbstractVector, a)
    return [ exp(-a*sum( (x-r).^2 )) for x in grid ]
end

density_tensor(grid; r=SVector(0.,0.,0.), a=1.) = density_tensor(grid, r, a)



function coulomb_tensor(ρ::AbstractArray, transtensor::AbtractTransformationTensor; tmax=nothing)
    V = similar(ρ)
    V .= 0
    v = similar(ρ)

    @showprogress "Calculating v-tensor..." for p in Iterators.reverse(eachindex(transtensor.wt))
        T = transtensor[:,:,:,:,p]
        @tensoropt v[α,β,γ,I,J,K] = T[α,α',I,I']*T[β,β',J,J']*T[γ,γ',K,K']*ρ[α',β',γ',I',J',K']
        V = V .+ transtensor.wt[p].*v
    end
    if tmax != nothing
        V = V .+ coulomb_correction(ρ, tmax)
    end
    return 2/sqrt(π).*V
end

function coulomb_tensor(ρ::AbstractArray, transtensor::AbstractArray, wt::AbstractVector; tmax=nothing)
    @assert size(transtensor)[end] == length(wt)
    V = similar(ρ)
    V .= 0
    v = similar(ρ)

    @showprogress "Calculating v-tensor..." for p in Iterators.reverse(eachindex(wt))
        T = @view transtensor[:,:,:,:,p]
        @tensoropt v[α,β,γ,I,J,K] = T[α,α',I,I']*T[β,β',J,J']*T[γ,γ',K,K']*ρ[α',β',γ',I',J',K']
        V = V .+ wt[p].*v
    end
    if tmax != nothing
        V = V .+ coulomb_correction(ρ, tmax)
    end
    return  2/sqrt(π).*V
end


coulomb_correction(ρ, tmax) = 2/sqrt(π)*(π/tmax^2).*ρ
