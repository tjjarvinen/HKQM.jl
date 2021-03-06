
abstract type AbtractTransformationTensor{T} <: AbstractArray{Float64, T} end

abstract type AbstractCoulombTransformation <: AbtractTransformationTensor{5} end

abstract type AbstractCoulombTransformationSingle{NT, NE, NG}  <: AbstractCoulombTransformation where {NT, NE, NG} end
abstract type AbstractCoulombTransformationCombination <: AbstractCoulombTransformation end
abstract type AbstractCoulombTransformationLocal{NT, NE, NG} <: AbstractCoulombTransformationSingle{NT, NE, NG} end


"""
    CoulombTransformation{T, NT, NE, NG} <: AbstractCoulombTransformationSingle{NT, NE, NG}

Coulomb transformation tensor with even spacing for t-points.

If `k` is zero this tensor solves Poisson equation. If nonzero it works as
Helmholz equation Greens function.

# Fields
- `elementgrid::SMatrix{NG,NE}` : Gauss points in elements
- `t::SVector{NT}{Float64}`     : t-coordinates
- `wt::Vector{T}`               : Weights for t-integration
- `w::SVector{NG}{Float64}`     : Weights for position coordinate integration
- `tmin::Float64`               : t-integration lower limit  (default=0.)
- `tmax::Float64`               : t-integration upper limit  (default=25.)
- `k::T`                        : Helmholz equation constant (default=0.)


# Contruction
    CoulombTransformation(ceg::CubicElementGrid, nt::Int; tmin=0., tmax=25., k=0.)


# Example
```jldoctest
julia> ceg = CubicElementGrid(5, 4, 16)
Cubic elements grid with 4^3 elements with 16^3 Gauss points

julia> CoulombTransformation(ceg, 48)
Coulomb transformation tensor for 48 t-points, tmin=0.0 tmax=25.0 and k=0.0

julia> ct = CoulombTransformation(ceg, 48; tmin=4., tmax=30., k=0.1)
Coulomb transformation tensor for 48 t-points, tmin=4.0 tmax=30.0 and k=0.1

julia> typeof(ct) <: AbstractArray{Float64,5}
true

julia> ct[3,3,1,1,5] ≈ 0.0158700408
true
```
"""
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


"""
    CoulombTransformationLog{T, NT, NE, NG} <: AbstractCoulombTransformationSingle{NT, NE, NG}

Coulomb transformation tensor with logarithmic spacing for t-points.

If `k` is zero this tensor solves Poisson equation. If nonzero it works as
Helmholz equation Greens function.

# Fields
- `elementgrid::SMatrix{NG,NE}` : Gauss points in elements
- `t::SVector{NT}{Float64}`     : t-coordinates
- `wt::Vector{T}`               : Weights for t-integration
- `w::SVector{NG}{Float64}`     : Weights for position coordinate integration
- `tmin::Float64`               : t-integration lower limit
- `tmax::Float64`               : t-integration upper limit
- `k::T`                        : Helmholz equation constant


# Construction
    CoulombTransformationLog(ceg::CubicElementGrid, nt::Int; kwargs...)

## Keywords
- `tmin=0.`
- `tmax=25.`
- `k=0.`


# Example
```jldoctest
julia> ceg = CubicElementGrid(5, 4, 16)
Cubic elements grid with 4^3 elements with 16^3 Gauss points

julia> CoulombTransformationLog(ceg, 48)
Logarithmic Coulomb transformation tensor for 48 t-points, tmin=0.0, tmax=25.0 and k=0.0

julia> ct = CoulombTransformationLog(ceg, 48; tmin=4., tmax=30., k=0.1)
Logarithmic Coulomb transformation tensor for 48 t-points, tmin=4.0, tmax=30.0 and k=0.1

julia> typeof(ct) <: AbstractArray{Float64,5}
true

julia> ct[3,3,1,1,5] ≈ 0.0594740698
true
```
"""
struct CoulombTransformationLog{T, NT, NE, NG} <: AbstractCoulombTransformationSingle{NT, NE, NG}
    elementgrid::SMatrix{NG,NE}
    t::SVector{NT}{Float64}
    wt::Vector{T}
    w::SVector{NG}{Float64}
    tmin::Float64
    tmax::Float64
    k::T
    function CoulombTransformationLog(ceg::CubicElementGrid, nt::Int;
                                      tmin=0., tmax=25., k=0.)
        s, ws = gausspoints(nt; elementsize=(log(tmin+1e-12), log(tmax)))
        t = exp.(s)
        wt = ws .* t
        grid = grid1d(ceg)
        ss = size(grid)
        wt = wt .* exp.(-(k^2)./(4t.^2))
        new{eltype(wt), nt, ss[2], ss[1]}(grid, t, wt, ceg.w, tmin, tmax, k)
    end
end


"""
    CoulombTransformationLocal{T, NT, NE, NG} <: AbstractCoulombTransformationSingle{NT, NE, NG}

Coulomb transformation tensor with even spacing for t-points and t-integration
local average correction.

If `k` is zero this tensor solves Poisson equation. If nonzero it works as
Helmholz equation Greens function.

# Fields
- `elementgrid::SMatrix{NG,NE}` : Gauss points in elements
- `t::SVector{NT}{Float64}`     : t-coordinates
- `wt::Vector{T}`               : Weights for t-integration
- `w::SVector{NG}{Float64}`     : Weights for position coordinate integration
- `tmin::Float64`               : t-integration lower limit
- `tmax::Float64`               : t-integration upper limit
- `k::T`                        : Helmholz equation constant
- `δ::Float64`                  : Correction constant for t-integration ∈ ]0,1]
- `δp::SMatrix{NG,NE}`          : Upper average points for t-integration correction
- `δm::SMatrix{NG,NE}`          : Lower average points for t-integration correction

# Construction
    CoulombTransformationLocal(ceg::CubicElementGrid, nt::Int; kwargs...)

## Keywords
- `tmin=0.`
- `tmax=25.`
- `δ=0.25`
- `k=0.`

# Example
```jldoctest
julia> ceg = CubicElementGrid(5, 4, 16)
Cubic elements grid with 4^3 elements with 16^3 Gauss points

julia> CoulombTransformationLocal(ceg, 48)
Coulomb transformation tensor with local correction for 48 t-points, tmin=0.0, tmax=25.0 and k=0.0

julia> ct = CoulombTransformationLocal(ceg, 48; tmin=4., tmax=30., k=0.1)
Coulomb transformation tensor with local correction for 48 t-points, tmin=4.0, tmax=30.0 and k=0.1

julia> typeof(ct) <: AbstractArray{Float64,5}
true

julia> ct[3,3,1,1,5] ≈ 0.0591078365
true
```
"""
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


"""
    CoulombTransformationLogLocal{T, NT, NE, NG} <: AbstractCoulombTransformationSingle{NT, NE, NG}

Coulomb transformation tensor with logarithmic spacing for t-points and
t-integration local average correction.

If `k` is zero this tensor solves Poisson equation. If nonzero it works as
Helmholz equation Greens function.

# Fields
- `elementgrid::SMatrix{NG,NE}` : Gauss points in elements
- `t::SVector{NT}{Float64}`     : t-coordinates
- `wt::Vector{T}`               : Weights for t-integration
- `w::SVector{NG}{Float64}`     : Weights for position coordinate integration
- `tmin::Float64`               : t-integration lower limit
- `tmax::Float64`               : t-integration upper limit
- `k::T`                        : Helmholz equation constant
- `δ::Float64`                  : Correction constant for t-integration ∈ ]0,1]
- `δp::SMatrix{NG,NE}`          : Upper average points for t-integration correction
- `δm::SMatrix{NG,NE}`          : Lower average points for t-integration correction

# Construction
    CoulombTransformationLogLocal(ceg::CubicElementGrid, nt::Int; kwargs...)

## Keywords
- `tmin=0.`
- `tmax=25.`
- `δ=0.25`
- `k=0.`

# Example
```jldoctest
julia> ceg = CubicElementGrid(5, 4, 16)
Cubic elements grid with 4^3 elements with 16^3 Gauss points

julia> CoulombTransformationLogLocal(ceg, 48)
Logarithmic Coulomb transformation tensor with local correction for 48 t-points, tmin=0.0, tmax=25.0 and k=0.0

julia> ct = CoulombTransformationLogLocal(ceg, 48; tmin=4., tmax=30., k=0.1)
Logarithmic Coulomb transformation tensor with local correction for 48 t-points, tmin=4.0, tmax=30.0 and k=0.1

julia> typeof(ct) <: AbstractArray{Float64,5}
true

julia> ct[3,3,1,1,5] ≈ 0.0591078365
true
```
"""
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
                                   tmin=0., tmax=25., δ=0.25, k=0.)
        @assert 0 < δ <= 1
        @assert 0 <= tmin < tmax
        s, ws = gausspoints(nt; elementsize=(log(tmin+1e-12), log(tmax)))
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


"""
    CoulombTransformationCombination{T, NE, NG} <: AbstractCoulombTransformationCombination

Coulomb transformation tensor that is composed of different types of tensors
with different t-integration ranges.

This tensor is intended to to be used to control how t-integration is performed.

# Fields
- `tensors::Vector{AbstractCoulombTransformation}` : Coulomb transformation subtensors
- `ref::Vector{Int}`  : index of tensor wich is referenced at given t-point index
- `t::Vector{Float64}`  : t-integration points
- `wt::Vector{T}`  : weights for t-integration
- `tmin::Float64`               : t-integration lower limit
- `tmax::Float64`               : t-integration upper limit
- `tindex::Vector{Int}`         : Index of t in subtensor
- `k::T`                        : Helmholz equation constant

# Construction
    CoulombTransformationCombination(t::AbstractCoulombTransformation...)


# Example
```jldoctest
julia> ceg = CubicElementGrid(5, 4, 16)
Cubic elements grid with 4^3 elements with 16^3 Gauss points

julia> ct1 = CoulombTransformationLog(ceg, 48)
Logarithmic Coulomb transformation tensor for 48 t-points, tmin=0.0, tmax=25.0 and k=0.0

julia> ct2 = CoulombTransformationLogLocal(ceg, 48; tmin=25, tmax=50)
Logarithmic Coulomb transformation tensor with local correction for 48 t-points, tmin=4.0, tmax=30.0 and k=0.1

julia> cc = CoulombTransformationCombination(ct1)
Coulomb transformation combination tensor for 48 t-points, tmin=0.0, tmax=25.0 and k=0.0

julia> push!(cc, ct2)
Coulomb transformation combination tensor for 96 t-points, tmin=0.0, tmax=50.0 and k=0.0

julia> CoulombTransformationCombination(ct1, ct2)
Coulomb transformation combination tensor for 96 t-points, tmin=0.0, tmax=50.0 and k=0.0
```
"""
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


function CoulombTransformationCombination(t::AbstractCoulombTransformationSingle...)
    ct = CoulombTransformationCombination(t[1])
    for x in t[2:end]
        push!(ct,x)
    end
    return ct
end

function loglocalct(ceg::CubicElementGrid, nt::Int; δ=0.25, tmax=300, tboundary=20, k=0.)
    @warn "loglocalct is depricated use optimal_coulomb_tranformation"
    @assert 0 < tboundary < tmax
    s, ws = gausspoints(nt; elementsize=(log(1e-12), log(tmax)))
    t = exp.(s)
    l = length(t[t .< tboundary])
    ct1 = CoulombTransformationLog(ceg, l; tmax=tboundary, k=k)
    ct2 = CoulombTransformationLogLocal(ceg, nt-l; tmin=tboundary, tmax=tmax, δ=δ, k=k)
    return CoulombTransformationCombination(ct1,ct2)
end

"""
    optimal_coulomb_tranformation(args...; kwargs...)

Returns Coulomb transformation tensor with optimal pre tested parameters.

# Args
- `ceg::CubicElementGrid`  :  grid where transformation is done
- `nt::Int`                :  number of t-points

# Keywords
- `δ=0.25`          : parameter for local average correction
- `tmax=700`        : maximum t value
- `tboundary=20`    : point after which local average correction is applied
- `k=0.`            : constant for Helmholz equation
"""
function optimal_coulomb_tranformation(ceg::CubicElementGrid, nt::Int; δ=0.25, tmax=700, tboundary=20, k=0.)
    @assert 0 < tboundary < tmax
    s, ws = gausspoints(nt; elementsize=(log(1e-12), log(tmax)))
    t = exp.(s)
    l = length(t[t .< tboundary])
    ct1 = CoulombTransformationLog(ceg, l; tmax=tboundary, k=k)
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
    s = size(cct)
    print(io, "Coulomb transformation tensor for $(s[end]) t-points,"*
              " tmin=$(cct.tmin), tmax=$(cct.tmax) and k=$(cct.k)"
    )
end

function Base.show(io::IO, ::MIME"text/plain", cct::CoulombTransformationLog)
    s = size(cct)
    print(io, "Logarithmic Coulomb transformation tensor for $(s[end]) t-points,"*
              " tmin=$(cct.tmin), tmax=$(cct.tmax) and k=$(cct.k)"
    )
end

function Base.show(io::IO, ::MIME"text/plain", cct::CoulombTransformationLocal)
    s = size(cct)
    print(io, "Coulomb transformation tensor with local correction for $(s[end])"*
              " t-points, tmin=$(cct.tmin), tmax=$(cct.tmax) and k=$(cct.k)"
    )
end

function Base.show(io::IO, ::MIME"text/plain", cct::CoulombTransformationCombination)
    s = size(cct)
    print(io, "Coulomb transformation combination tensor for $(s[end])"*
              " t-points, tmin=$(cct.tmin), tmax=$(cct.tmax) and k=$(cct.k)"
    )
end

function Base.show(io::IO, ::MIME"text/plain", cct::CoulombTransformationLogLocal)
    s = size(cct)
    print(io, "Logarithmic Coulomb transformation tensor with local correction for $(s[end])"*
              " t-points, tmin=$(cct.tmin), tmax=$(cct.tmax) and k=$(cct.k)"
    )
end

## Integration weight tensor

"""
    ω_tensor(ceg::CubicElementGrid) -> Matrix{Float64}

Return a tensor that contains integration constants to perform integral over
element grid.
"""
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

"""
    DerivativeTensor{NG} <: AbstractDerivativeTensor

Tensor that performs derivate over [`CubicElementGrid`](@ref).

# Fields
- `gauss_points::SVector{NG}{Float64}`  : point for integration
- `values::Matrix{Float64}`             : tensor values

# Construction
    DerivativeTensor(ceg::CubicElementGrid)
"""
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
        # Derivative is done by analytically deriving Gauss-Legendre basis
        # functions and then calculating values over the derivatives.
        # f(x) = ∑aᵢpᵢ(x) → f'(x) = ∑aᵢpᵢ'(x)
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
    @warn "kinetic_energy is deprecated use operators instead"
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
        @warn "MomentumTensor is deprecated use operators instead"
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


## Nuclear potential tensors

abstract type AbstractNuclearPotential <: AbtractTransformationTensor{3} end
abstract type AbstractNuclearPotentialSingle{T} <: AbstractNuclearPotential end
abstract type AbstractNuclearPotentialCombination{T} <: AbstractNuclearPotential end


"""
    NuclearPotentialTensor{T}

Tensor for nuclear potential in 1D. To get 3D you need 3 of these tensors and
integrate t-coordinate.

Easy way is to use [`PotentialTensor`](@ref) to get 3D tensor.

# Fields
- `elementgrid::Matrix{T}`   : Grid definition, collums are elements, rows Gauss points
- `t::Vector{Float64}`       : t-coordinate
- `wt::Vector{Float64}`      : Integration weights for t
- `tmin::Float64`            : Minimum t-value
- `tmax::Float64`            : Maximum t-value
- `r::T`                     : Nuclear coordinate

# Creation
    NuclearPotentialTensor(r, ceg::CubicElementGrid, nt::Int; tmin=0, tmax=30)

- `r`                       : Nuclear coordinate (1D)
- `ceg::CubicElementGrid`   : Grid for the potential
- `nt::Int`                 : Number of t-points
"""
struct NuclearPotentialTensor{T} <: AbstractNuclearPotentialSingle{T}
    elementgrid::Matrix{T}
    t::Vector{Float64}
    wt::Vector{Float64}
    tmin::Float64
    tmax::Float64
    r::T
    function NuclearPotentialTensor(r, ceg::CubicElementGrid, nt::Int; tmin=0, tmax=30)
        @assert dimension(r) == NoDims || dimension(r) == dimension(u"m")
        rt = austrip(r)
        t, wt = gausspoints(nt; elementsize=(tmin, tmax))
        grid = Array(grid1d(ceg)) .- rt
        new{eltype(grid)}(grid, t, wt, tmin, tmax, rt)
    end
end

"""
    NuclearPotentialTensorLog{T}

Tensor for nuclear potential in 1D. To get 3D you need 3 of these tensors and
integrate t-coordinate. This version has logarithmic spacing for t-points.

Easy way is to use [`PotentialTensor`](@ref) to get 3D tensor.

# Fields
- `elementgrid::Matrix{T}`   : Grid definition, collums are elements, rows Gauss points
- `t::Vector{Float64}`       : t-coordinate
- `wt::Vector{Float64}`      : Integration weights for t
- `tmin::Float64`            : Minimum t-value
- `tmax::Float64`            : Maximum t-value
- `r::T`                     : Nuclear coordinate

# Creation
    NuclearPotentialTensorLog(r, ceg::CubicElementGrid, nt::Int; tmin=0, tmax=30)

- `r`                       : Nuclear coordinate (1D)
- `ceg::CubicElementGrid`   : Grid for the potential
- `nt::Int`                 : Number of t-points
"""
struct NuclearPotentialTensorLog{T} <: AbstractNuclearPotentialSingle{T}
    elementgrid::Matrix{T}
    t::Vector{Float64}
    wt::Vector{Float64}
    tmin::Float64
    tmax::Float64
    r::T
    function NuclearPotentialTensorLog(r, ceg::CubicElementGrid, nt::Int; tmin=0, tmax=30)
        @assert dimension(r) == NoDims || dimension(r) == dimension(u"m")
        rt = austrip(r)
        s, ws = gausspoints(nt; elementsize=(log(tmin+1e-12), log(tmax)))
        t = exp.(s)
        wt = ws .* t
        grid = Array(grid1d(ceg)) .- rt
        new{eltype(grid)}(grid, t, wt, tmin, tmax, rt)
    end
end


"""
    NuclearPotentialTensorLogLocal{T}

Tensor for nuclear potential in 1D. To get 3D you need 3 of these tensors and
integrate t-coordinate. This version has logarithmic spacing for t-points and
local average correction.

Easy way is to use [`PotentialTensor`](@ref) to get 3D tensor.

# Fields
- `elementgrid::Matrix{T}`   : Grid definition, collums are elements, rows Gauss points
- `t::Vector{Float64}`       : t-coordinate
- `wt::Vector{Float64}`      : Integration weights for t
- `tmin::Float64`            : Minimum t-value
- `tmax::Float64`            : Maximum t-value
- `r::T`                     : Nuclear coordinate
- `δ::Float64`               : Correction parameter ∈[0,1]
- `δp::Matrix{T}`            : Upper limits for correction
- `δm::Matrix{T}`            : Lower limits for correction

# Creation
    NuclearPotentialTensorLogLocal(r, ceg::CubicElementGrid, nt::Int; tmin=0, tmax=30, δ=0.25)

- `r`                       : Nuclear coordinate (1D)
- `ceg::CubicElementGrid`   : Grid for the potential
- `nt::Int`                 : Number of t-points
"""
struct NuclearPotentialTensorLogLocal{T} <: AbstractNuclearPotentialSingle{T}
    elementgrid::Matrix{T}
    t::Vector{Float64}
    wt::Vector{Float64}
    tmin::Float64
    tmax::Float64
    r::T
    δ::Float64
    δp::Matrix{T}
    δm::Matrix{T}
    function NuclearPotentialTensorLogLocal(r, ceg::CubicElementGrid, nt::Int;
                                           tmin=0, tmax=30, δ=0.25)
        @assert 0 < δ <= 1
        @assert 0 <= tmin < tmax
        @assert dimension(r) == NoDims || dimension(r) == dimension(u"m")
        rt = austrip(r)
        s, ws = gausspoints(nt; elementsize=(log(tmin+1e-12), log(tmax)))
        t = exp.(s)
        wt = ws .* t
        t, wt = gausspoints(nt; elementsize=(tmin, tmax))
        grid = Array(grid1d(ceg)) .- rt
        ss = size(grid)

        # get range around points [x+δm, x+δp]
        # and calculate average on that range (later on)
        δp = zeros(ss...)
        δm = zeros(ss...)
        δm[2:end,:] = grid[1:end-1,:] .- grid[2:end,:]
        δp[1:end-1,:] = grid[2:end,:] .- grid[1:end-1,:]
        for j in 1:ss[2]
            δm[1,j] = ceg.elements[j].low - grid[1,j]
            δp[end,j] = ceg.elements[j].high - grid[end,j]
        end
        new{eltype(δm)}(grid, t, wt, tmin, tmax, rt, δ, δ.*δp, δ.*δm)
    end
end

"""
    NuclearPotentialTensorGaussian{T}

Tensor for nuclear potential in 1D. To get 3D you need 3 of these tensors and
integrate t-coordinate. This version has logarithmic spacing for t-points
with gaussian nuclear density.

Easy way is to use [`PotentialTensor`](@ref) to get 3D tensor.

# Fields
- `elementgrid::Matrix{T}`   : Grid definition, collums are elements, rows Gauss points
- `t::Vector{Float64}`       : t-coordinate
- `wt::Vector{Float64}`      : Integration weights for t
- `tmin::Float64`            : Minimum t-value
- `tmax::Float64`            : Maximum t-value
- `r::T`                     : Nuclear coordinate
- `β::Float64`               : Width of Gaussian exp(-βr²)

# Creation
    NuclearPotentialTensorGaussian(r, ceg::CubicElementGrid, nt; β=100, tmin=0, tmax=20)

- `r`                       : Nuclear coordinate (1D)
- `ceg::CubicElementGrid`   : Grid for the potential
- `nt::Int`                 : Number of t-points
"""
struct NuclearPotentialTensorGaussian{T} <: AbstractNuclearPotentialSingle{T}
    elementgrid::Matrix{T}
    t::Vector{Float64}
    wt::Vector{Float64}
    tmin::Float64
    tmax::Float64
    r::T
    β::Float64
    function NuclearPotentialTensorGaussian(r, ceg::CubicElementGrid, nt, β; tmin=0, tmax=20)
        @assert dimension(r) == NoDims || dimension(r) == dimension(u"m")
        rt = austrip(r)
        s, ws = gausspoints(nt; elementsize=(log(tmin+1e-12), log(tmax)))
        t = exp.(s)
        wt = ws .* t
        grid = Array(grid1d(ceg)) .- rt
        new{eltype(grid)}(grid, t, wt, tmin, tmax, rt, β)
    end
end

function NuclearPotentialTensorGaussian(r, ceg::CubicElementGrid, nt; σ=0.1, tmin=0, tmax=20)
    min_d = elementsize(ceg.elements) / size(ceg)[1]
    β = 0.5*(σ*min_d)^-2
    return NuclearPotentialTensorGaussian(r, ceg, nt, β; tmin=tmin, tmax=tmax)
end

"""
    NuclearPotentialTensorCombination{T}

Combine different tensors to bigger one. Allows combining normal, logarithmic
and local correction ones to one tensor.

# Fields
- `subtensors::Vector{AbstractNuclearPotentialSingle{T}}`  :  combination of these
- `ref::Vector{Int}`    :  transforms t-index to subtensor index
- `index::Vector{Int}`  :  transforms t-index to subtensors t-index
- `wt::Vector{Float64}` :  t-integration weight
- `tmin::Float64`       :  minimum t-value
- `tmax::Float64`       :  maximum t-value
- `r::T`                :  nuclear coordinate

# Creation
    NuclearPotentialTensorCombination(npt::AbstractNuclearPotentialSingle{T}...)
"""
mutable struct NuclearPotentialTensorCombination{T} <: AbstractNuclearPotentialCombination{T}
    subtensors::Vector{AbstractNuclearPotentialSingle{T}}
    ref::Vector{Int}
    index::Vector{Int}
    t::Vector{Float64}
    wt::Vector{Float64}
    tmin::Float64
    tmax::Float64
    r::T
    function NuclearPotentialTensorCombination(npt::AbstractNuclearPotentialSingle)
        nt = size(npt)[end]
        new{eltype(npt.elementgrid)}([npt],
                   ones(nt),
                   1:nt,
                   deepcopy(npt.t),
                   deepcopy(npt.wt),
                   npt.tmin,
                   npt.tmax,
                   npt.r
                   )
    end
end

function NuclearPotentialTensorCombination(npt::AbstractNuclearPotentialSingle{T}...) where {T}
    tmp = NuclearPotentialTensorCombination(npt[1])
    for x in npt[2:end]
        push!(tmp, x)
    end
    return tmp
end


function Base.show(io::IO, ::MIME"text/plain", npt::AbstractNuclearPotential)
    print(io, "Nuclear potential tensor for $(length(npt.t)) t-points,"*
              " tmin=$(npt.tmin), tmax=$(npt.tmax)"
    )
end


Base.size(npt::AbstractNuclearPotential) = (size(npt.elementgrid)..., length(npt.t))

function Base.size(npct::NuclearPotentialTensorCombination)
    return (size(npct.subtensors[1])[1:2]..., length(npct.t))
end


function Base.getindex(npt::AbstractNuclearPotential, i::Int, j::Int, p::Int)
    r = npt.elementgrid[i,j]
    return exp(-npt.t[p]^2 * r^2)
end


function Base.getindex(npt::NuclearPotentialTensorLogLocal, i::Int, j::Int, p::Int)
    r = npt.elementgrid[i,j]
    δ = abs(npt.δp[i,j] - npt.δm[i,j])
    rmax = (r+δ) * npt.t[p]
    rmin = (r-δ) * npt.t[p]
    return 0.5 * √π * erf(rmin, rmax) / (rmax-rmin)
end

function Base.getindex(npt::NuclearPotentialTensorGaussian, i::Int, j::Int, p::Int)
    r = npt.elementgrid[i,j]
    t = npt.t[p]
    a = npt.β
    q = a * t^2 / (t^2 + a)
    return sqrt(a/(t^2+a)) * exp(-q * r^2)
end


function Base.getindex(npct::NuclearPotentialTensorCombination, i::Int, j::Int, p::Int)
    s = npct.ref[p]
    t = npct.index[p]
    return npct.subtensors[s][i,j,t]
end


function Base.push!(nptc::NuclearPotentialTensorCombination{T},
                    npt::AbstractNuclearPotentialSingle{T}) where {T}
    @assert nptc.r == npt.r
    @assert nptc.tmax <= npt.tmin
    push!(nptc.subtensors, npt)
    n = size(npt)[end]
    append!(nptc.ref, (length(nptc.subtensors))*ones(n))
    append!(nptc.t, npt.t)
    append!(nptc.wt, npt.wt)
    append!(nptc.index, 1:n)
    nptc.tmax = npt.tmax
    return nptc
end


"""
    PotentialTensor{Tx,Ty,Tz}

Used to create nuclear potential.

# Creation
    PotentialTensor(npx, npy, npz)

`npx`, `npy` and `npz` are nuclear potential tensors.
To get the best result usually they should be [`NuclearPotentialTensorCombination`](@ref).

# Usage
Create tensor and then call `Array` on it to get nuclear potential.

"""
struct PotentialTensor{Tx<:AbstractNuclearPotential,
                       Ty<:AbstractNuclearPotential,
                       Tz<:AbstractNuclearPotential}
    x::Tx
    y::Ty
    z::Tz
    function PotentialTensor(npx, npy, npz)
        @assert size(npx)[end] == size(npy)[end] == size(npz)[end]
        @assert npx.tmin == npy.tmin == npx.tmin
        @assert npx.tmax == npy.tmax == npx.tmax
        new{typeof(npx), typeof(npy), typeof(npz)}(npx, npy, npz)
    end
end

Base.show(io::IO, ::MIME"text/plain", pt::PotentialTensor) = print(io, "PotentialTensor")

function Base.size(pt::PotentialTensor)
    s1 = size(pt.x)
    s2 = size(pt.y)
    s3 = size(pt.z)
    return s1[1], s2[1], s3[1], s1[2], s2[2], s3[2]
end


function Base.getindex(pt::PotentialTensor, i::Int, j::Int, k::Int, I::Int, J::Int, K::Int)
    out = pt.x[i,I,:]
    out .*= pt.y[j,J,:]
    out .* pt.z[k,K,:]
    out .* pt.x.wt
    return out
end

function Base.Array(pt::PotentialTensor)
    x = Array(pt.x)
    y = Array(pt.y)
    z = Array(pt.z)
    @tullio out[i,j,k,I,J,K] := x[i,I,t] * y[j,J,t] * z[k,K,t] * pt.x.wt[t]
    return 2/sqrt(π) .* out
end


"""
    nuclear_potential(ceg::CubicElementGrid, q, r)
    nuclear_potential(ceg::CubicElementGrid, aname::String, r)

Gives nuclear potential for given nuclear charge or atom name.
"""
function nuclear_potential(ceg::CubicElementGrid, q, r)

    ncx = optimal_nuclear_tensor(ceg, r[1])
    ncy = optimal_nuclear_tensor(ceg, r[2])
    ncz = optimal_nuclear_tensor(ceg, r[3])

    return nuclear_potential(ceg, q, ncx, ncy, ncz)
end

function optimal_nuclear_tensor(ceg, x)
    np = NuclearPotentialTensor(x, ceg, 64; tmin=0, tmax=70)
    npl = NuclearPotentialTensorLog(x, ceg, 16; tmin=70, tmax=120)
    npll = NuclearPotentialTensorGaussian(x, ceg, 16;  σ=0.1, tmin=120, tmax=300)
    nplll = NuclearPotentialTensorGaussian(x, ceg, 32; σ=0.1, tmin=300, tmax=20000)
    return NuclearPotentialTensorCombination(np, npl, npll, nplll)
end

function nuclear_potential(ceg,
                           q,
                           npx::AbstractNuclearPotential,
                           npy::AbstractNuclearPotential,
                           npz::AbstractNuclearPotential)
    @assert dimension(q) == dimension(u"C") || dimension(q) == NoDims
    pt = PotentialTensor(npx, npy, npz)
    qau = austrip(q)
    return (qau*u"hartree/e_au") * ScalarOperator(ceg, Array(pt))
end


function nuclear_potential(ceg::CubicElementGrid, aname::String, r)
    if length(aname) <= 2    # is atomic symbol
        e = elements[Symbol(aname)]
    else
        e = elements[aname]
    end
    @debug "element number $(e.number)"
    return nuclear_potential(ceg, e.number, r)
end
