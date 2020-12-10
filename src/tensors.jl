
abstract type AbtractTransformationTensor{T} <: AbstractArray{Float64, T} end

abstract type AbstractCoulombTransformation <: AbtractTransformationTensor{5} end

abstract type AbstractCoulombTransformationSingle{NT, NE, NG}  <: AbstractCoulombTransformation where {NT, NE, NG} end
abstract type AbstractCoulombTransformationCombination <: AbstractCoulombTransformation end
abstract type AbstractCoulombTransformationLocal{NT, NE, NG} <: AbstractCoulombTransformationSingle{NT, NE, NG} end

struct CoulombTransformation{T, NT, NE, NG} <: AbstractCoulombTransformationSingle{NT, NE, NG}
    elementgrid::SMatrix{NG,NE}
    t::SVector{NT}{Float64}
    wt::Vector{T}
    w::SVector{NG}{Float64}
    tmin::Float64
    tmax::Float64
    k::T
    function CoulombTransformation(ceg::CubicElementGrid, nt::Int; tmin=0., tmax=25., k=0.)
        t, wt = gausspoints(nt; elementsize=(tmin, tmax))
        grid = grid1d(ceg)
        s = size(grid)
        wt = wt .* exp.(-(k^2)./(4t.^2))
        new{eltype(wt), nt, s[2], s[1]}(grid, t, wt, ceg.w, tmin, tmax, k)
    end
end

struct CoulombTransformationLog{T, NT, NE, NG} <: AbstractCoulombTransformationSingle{NT, NE, NG}
    elementgrid::SMatrix{NG,NE}
    t::SVector{NT}{Float64}
    wt::Vector{T}
    w::SVector{NG}{Float64}
    tmin::Float64
    tmax::Float64
    k::T
    function CoulombTransformationLog(ceg::CubicElementGrid, nt::Int;
                                   tmin=0., tmax=25., ε=1e-12, k=0.)
        s, ws = gausspoints(nt; elementsize=(log(tmin+ε), log(tmax)))
        t = exp.(s)
        wt = ws .* t
        grid = grid1d(ceg)
        ss = size(grid)
        wt = wt .* exp.(-(k^2)./(4t.^2))
        new{eltype(wt), nt, ss[2], ss[1]}(grid, t, wt, ceg.w, tmin, tmax, k)
    end
end

struct CoulombTransformationLocal{T, NT, NE, NG} <: AbstractCoulombTransformationLocal{NT, NE, NG}
    elementgrid::SMatrix{NG,NE}
    t::SVector{NT}{Float64}
    wt::Vector{T}
    w::SVector{NG}{Float64}
    tmin::Float64
    tmax::Float64
    k::T
    δ::Float64
    δp::SMatrix{NG,NE}
    δm::SMatrix{NG,NE}
    function CoulombTransformationLocal(ceg::CubicElementGrid, nt::Int;
                                   tmin=0., tmax=25., δ=0.25, k=0.)
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
        wt = wt .* exp.(-(k^2)./(4t.^2))
        new{eltype(wt), nt, s[2], s[1]}(grid, t, wt, ceg.w, tmin, tmax, k, δ, δ.*δp, δ.*δm)
    end
end

struct CoulombTransformationLogLocal{T, NT, NE, NG} <: AbstractCoulombTransformationLocal{NT, NE, NG}
    elementgrid::SMatrix{NG,NE}
    t::SVector{NT}{Float64}
    wt::Vector{T}
    w::SVector{NG}{Float64}
    tmin::Float64
    tmax::Float64
    k::T
    δ::Float64
    δp::SMatrix{NG,NE}
    δm::SMatrix{NG,NE}
    function CoulombTransformationLogLocal(ceg::CubicElementGrid, nt::Int;
                                   tmin=0., tmax=25., δ=0.25, ε=1e-12, k=0.)
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
        wt = wt .* exp.(-(k^2)./(4t.^2))
        new{eltype(wt), nt, ss[2], ss[1]}(grid, t, wt, ceg.w, tmin, tmax, k, δ, δ.*δp, δ.*δm)
    end
end

mutable struct CoulombTransformationCombination{T, NE, NG} <: AbstractCoulombTransformationCombination
    tensors::Vector{AbstractCoulombTransformation}
    ref::Vector{Int}
    t::Vector{Float64}
    wt::Vector{T}
    tmin::Float64
    tmax::Float64
    tindex::Vector{Int}
    k::T
    function CoulombTransformationCombination(ct::AbstractCoulombTransformationSingle)
        s = size(ct)
        new{eltype(ct.wt), s[3], s[2]}([ct], ones(s[end]), ct.t, ct.wt, ct.tmin, ct.tmax, 1:length(ct.t), ct.k)
    end
end

function CoulombTransformationCombination(t...)
    ct = CoulombTransformationCombination(t[1])
    for x ∈ t[2:end]
        push!(ct,x)
    end
    return ct
end

function loglocalct(ceg::CubicElementGrid, nt::Int; δ=0.25, tmax=300, tboundary=20, ε=1e-12, k=0.)
    @assert 0 < tboundary < tmax
    s, ws = gausspoints(nt; elementsize=(log(ε), log(tmax)))
    t = exp.(s)
    l = length(t[t .< tboundary])
    ct1 = CoulombTransformationLog(ceg, l; tmax=tboundary, ε=ε, k=k)
    ct2 = CoulombTransformationLogLocal(ceg, nt-l; tmin=tboundary, tmax=tmax, δ=δ, k=k)
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

function Base.size(ct::CoulombTransformationCombination{T,NE,NG}) where {T,NE,NG}
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

function  Base.push!(ctc::CoulombTransformationCombination{T,NE,NG},
            tt::AbstractCoulombTransformationSingle{NT,NE,NG}) where {T,NT,NE,NG}
    @assert ctc.tensors[end].tmax <= tt.tmin
    @assert ctc.k == tt.k
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

## Integration weight tensor

function ω_tensor(ceg::CubicElementGrid)
    n = size(ceg)[end]
    l = length(ceg.w)
    ω = reshape( repeat(ceg.w, n), l, n )
end

## Help functions to create charge density

function density_tensor(grid::AbstractArray, r::AbstractVector, a)
    return [ exp(-a*sum( (x-r).^2 )) for x in grid ]
end

density_tensor(grid; r=SVector(0.,0.,0.), a=1.) = density_tensor(grid, r, a)



## Derivative tensor

abstract type AbstractDerivativeTensor <: AbtractTransformationTensor{2} end


struct DerivativeTensor{NG} <: AbstractDerivativeTensor
    gauss_points::SVector{NG}{Float64}
    values::Matrix{Float64}
    function DerivativeTensor(ceg::CubicElementGrid)
        function _lpoly(x)
            # Legendre polynomials (without prefactor)
            l = length(x)
            out = []
            for i in 1:l
                z = vcat(x[1:i-1],x[i+1:end])
                push!(out, fromroots(z))
            end
            return out
        end
        function _pre_a(x)
            # Legendre plynomial prefactors
            l = length(x)
            a = ones(l)
            for i in 1:l
                for j in 1:l
                    if j != i
                        a[i] *= x[i] - x[j]
                    end
                end
            end
            return a.^-1
        end
        x = BigFloat.(ceg.gpoints)
        xmin=-0.5*elementsize(ceg.elements)
        xmax=-xmin
        grid = grid1d(ceg)
        ng, ne = size(grid)
        a=_pre_a(x)
        p=_lpoly(x)
        pd = derivative.(p)
        ϕ = zeros(length(x), length(pd))
        for i in eachindex(pd)
            ϕ[:,i] = a[i]*pd[i].(x)
        end
        new{ng}(x, ϕ)
    end
end


function Base.getindex(dt::DerivativeTensor, i::Int, j::Int)
    return dt.values[i,j]
end

Base.size(dt::DerivativeTensor) = size(dt.values)


function Base.show(io::IO, ::MIME"text/plain", dt::DerivativeTensor)
    print(io, "Derivative tensor size=$(size(dt))")
end


function kinetic_energy(ceg::CubicElementGrid, dt, ψ)
    tmp = similar(ψ)

    @tensor tmp[i,j,k,I,J,K] = dt[i,l] * ψ[l,j,k,I,J,K]
    ex = 0.5*integrate(tmp, ceg, tmp)

    @tensor tmp[i,j,k,I,J,K] = dt[j,l] * ψ[i,l,k,I,J,K]
    ey = 0.5*integrate(tmp, ceg, tmp)

    @tensor tmp[i,j,k,I,J,K] = dt[k,l] * ψ[i,j,l,I,J,K]
    ez = 0.5*integrate(tmp, ceg, tmp)

    return ex+ey+ez
end



## Momentum tensor

struct MomentumTensor{NG} <: AbstractArray{ComplexF64, 3}
    dt::DerivativeTensor{NG}
    function MomentumTensor(dt::DerivativeTensor{T}) where T
        new{T}(dt)
    end
end

function MomentumTensor(ceg::CubicElementGrid)
    return MomentumTensor(DerivativeTensor(ceg))
end

Base.size(mt::MomentumTensor) = (size(mt.dt)...,3)

Base.getindex(mt::MomentumTensor,i::Int,j::Int,xyz::Int) = -ComplexF64(0,mt.dt[i,j])

function Base.show(io::IO, ::MIME"text/plain", mt::MomentumTensor)
    print(io, "Momentum tensor size=$(size(mt))")
end
