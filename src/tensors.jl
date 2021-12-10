



"""
    HelmholtzTensorLinear{T} <: AbstractHelmholtzTensorSingle

Helmholtz/Poisson equation tensor with even spacing for t-points.

If `k` is zero this tensor solves Poisson equation. If nonzero it works as
Helmholz equation Greens function.

# Fields
- `elementgrid::AbstractElementGrid{<:Any, 2}` : Gauss points and elements in 1D
- `t::Vector{Float64}`                         : t-coordinates
- `wt::Vector{T}`                              : Weights for t-integration
- `w::Matrix{Float64}`          : Weights for position coordinate integration
- `tmin::Float64`               : t-integration lower limit  (default=0.)
- `tmax::Float64`               : t-integration upper limit  (default=25.)
- `k::T`                        : Helmholz equation constant (default=0.)


# Contruction
    HelmholtzTensorLinear(ceg::AbstractElementGridSymmetricBox, nt::Int; tmin=0., tmax=25., k=0.)


# Example
```jldoctest
julia> ceg = CubicElementGrid(5u"Å", 4, 16)
Cubic elements grid with 4^3 elements with 16^3 Gauss points

julia> HelmholtzTensorLinear(ceg, 48)
HelmholtzTensorLinear 4 elements and 16 Gauss points per element

julia> ht = HelmholtzTensorLinear(ceg, 48; tmin=4., tmax=30., k=0.1)
HelmholtzTensorLinear 4 elements and 16 Gauss points per element

julia> typeof(ht) <: AbstractArray{Float64,5}
true

julia> ht[3,3,1,1,5] ≈ 0.1123897034
true
```
"""
struct HelmholtzTensorLinear{T} <: AbstractHelmholtzTensorSingle
    elementgrid::AbstractElementGrid{<:Any, 2}
    t::Vector{Float64}
    wt::Vector{T}
    w::Matrix{Float64}
    tmin::Float64
    tmax::Float64
    k::T
    function HelmholtzTensorLinear(eg::AbstractElementGridSymmetricBox, nt::Int; tmin=0., tmax=25., k=0.)
        t, wt = gausspoints(nt; elementsize=(tmin, tmax))
        elementgrid = get_1d_grid(eg)
        w = getweight(elementgrid)
        s = size(w)
        wt = wt .* exp.(-(k^2)./(4t.^2))
        new{typeof(k*t[1])}(elementgrid, t, wt, w, tmin, tmax, k)
    end

end


function Base.size(ct::HelmholtzTensorLinear)
    s = size(ct.elementgrid)
    st = length(ct.t)
    return (s[1], s[1], s[2], s[2], st)
end

function Base.getindex(ct::HelmholtzTensorLinear, i::Int, j::Int, I::Int, J::Int, tt::Int)
    r = ct.elementgrid[i,I] - ct.elementgrid[j,J]
    return exp(-(ct.t[tt]*r)^2) * ct.w[j,J]
end

function Base.show(io::IO, ::MIME"text/plain", ct::HelmholtzTensorLinear)
    s = size(ct)
    print(io, "HelmholtzTensorLinear $(s[3]) elements and $(s[1]) Gauss points per element")
end


"""
    HelmholtzTensorLog{T} <: AbstractHelmholtzTensorSingle

Helmholtz/Poisson equation tensor with logarithmic spacing for t-points.

If `k` is zero this tensor solves Poisson equation. If nonzero it works as
Helmholz equation Greens function.

# Fields
- `elementgrid::AbstractElementGrid{<:Any, 2}` : Gauss points and elements in 1D
- `t::Vector{Float64}`                         : t-coordinates
- `wt::Vector{T}`                              : Weights for t-integration
- `w::Matrix{Float64}`          : Weights for position coordinate integration
- `tmin::Float64`               : t-integration lower limit  (default=0.)
- `tmax::Float64`               : t-integration upper limit  (default=25.)
- `k::T`                        : Helmholz equation constant (default=0.)


# Construction
    HelmholtzTensorLog(ceg::AbstractElementGridSymmetricBox, nt::Int; kwargs...)

## Keywords
- `tmin=0.`
- `tmax=25.`
- `k=0.`


# Example
```jldoctest
julia> ceg = CubicElementGrid(5u"Å", 4, 16)
Cubic elements grid with 4^3 elements with 16^3 Gauss points

julia> HelmholtzTensorLog(ceg, 48)
HelmholtzTensorLog 4 elements and 16 Gauss points per element

julia> ht = HelmholtzTensorLog(ceg, 48; tmin=4., tmax=30., k=0.1)
HelmholtzTensorLog 4 elements and 16 Gauss points per element

julia> typeof(ht) <: AbstractArray{Float64,5}
true

julia> ht[3,3,1,1,5] ≈ 0.1123897034
true
```
"""
struct HelmholtzTensorLog{T} <: AbstractHelmholtzTensorSingle
    elementgrid::AbstractElementGrid{<:Any, 2}
    t::Vector{Float64}
    wt::Vector{T}
    w::Matrix{Float64}
    tmin::Float64
    tmax::Float64
    k::T
    function HelmholtzTensorLog(eg::AbstractElementGridSymmetricBox, nt::Int;
                                         tmin=0., tmax=25., k=0.)
        s, ws = gausspoints(nt; elementsize=(log(tmin+1e-12), log(tmax)))
        t = exp.(s)
        wt = ws .* t
        elementgrid = get_1d_grid(eg)
        w = getweight(elementgrid)
        s = size(w)
        wt = wt .* exp.(-(k^2)./(4t.^2))
        new{typeof(k*t[1])}(elementgrid, t, wt, w, tmin, tmax, k)
    end
end


function Base.size(ct::HelmholtzTensorLog)
    s = size(ct.elementgrid)
    st = length(ct.t)
    return (s[1], s[1], s[2], s[2], st)
end

function Base.getindex(ct::HelmholtzTensorLog, i::Int, j::Int, I::Int, J::Int, tt::Int)
    r = ct.elementgrid[i,I] - ct.elementgrid[j,J]
    return exp(-(ct.t[tt]*r)^2) * ct.w[j,J]
end

function Base.show(io::IO, ::MIME"text/plain", ct::HelmholtzTensorLog)
    s = size(ct)
    print(io, "HelmholtzTensorLog $(s[3]) elements and $(s[1]) Gauss points per element")
end


"""
    HelmholtzTensorLocalLinear{T} <: AbstractHelmholtzTensorSingle

Helmholtz/Poisson equation tensor with even spacing for t-points and t-integration
local average correction.

If `k` is zero this tensor solves Poisson equation. If nonzero it works as
Helmholz equation Greens function.

# Fields
- `elementgrid::AbstractElementGrid{<:Any, 2}` : Gauss points and elements
- `t::Vector{Float64}`          : t-coordinates
- `wt::Vector{T}`               : Weights for t-integration
- `w::Matrix{Float64}`          : Weights for position coordinate integration
- `tmin::Float64`               : t-integration lower limit
- `tmax::Float64`               : t-integration upper limit
- `k::T`                        : Helmholz equation constant
- `δ::Float64`                  : Correction constant for t-integration ∈ ]0,1]
- `δp::Matrix{Float64}`         : Upper average points for t-integration correction
- `δm::Matrix{Float64}`         : Lower average points for t-integration correction

# Construction
    HelmholtzTensorLocalLinear(ceg::CubicElementGrid, nt::Int; kwargs...)

## Keywords
- `tmin=0.`
- `tmax=25.`
- `δ=0.25`
- `k=0.`

# Example
```jldoctest
julia> ceg = CubicElementGrid(5u"Å", 4, 16)
Cubic elements grid with 4^3 elements with 16^3 Gauss points

julia> HelmholtzTensorLocalLinear(ceg, 48)
HelmholtzTensorLocalLinear 4 elements and 16 Gauss points per element

julia> ht = HelmholtzTensorLocalLinear(ceg, 48; tmin=4., tmax=30., k=0.1)
HelmholtzTensorLocalLinear 4 elements and 16 Gauss points per element

julia> typeof(ht) <: AbstractArray{Float64,5}
true

julia> ht[3,3,1,1,5] ≈ 0.10995313868
true
```
"""
struct HelmholtzTensorLocalLinear{T} <: AbstractHelmholtzTensorSingle
    elementgrid::AbstractElementGrid{<:Any, 2}
    t::Vector{Float64}
    wt::Vector{T}
    w::Matrix{Float64}
    tmin::Float64
    tmax::Float64
    k::T
    δ::Float64
    δp::Matrix{Float64}
    δm::Matrix{Float64}
    function HelmholtzTensorLocalLinear(eg::AbstractElementGridSymmetricBox, nt::Int;
                                   tmin=0., tmax=25., δ=0.25, k=0.)
        @assert 0 < δ <= 1
        @assert 0 <= tmin < tmax
        t, wt = gausspoints(nt; elementsize=(tmin, tmax))
        grid = get_1d_grid(eg)
        s = size(grid)
        δp = zeros(s...)
        δm = zeros(s...)
        δm[2:end,:] = grid[1:end-1,:] .- grid[2:end,:]
        δp[1:end-1,:] = grid[2:end,:] .- grid[1:end-1,:]
        for j in axes(grid, 2)
            δm[1,j] =  austrip( getelement(grid, j).low ) - grid[1,j]
            δp[end,j] = austrip( getelement(grid, j).high ) - grid[end,j]
        end
        wt = wt .* exp.(-(k^2)./(4t.^2))
        w = getweight(grid)
        new{eltype(wt)}(grid, t, wt, w, tmin, tmax, k, δ, δ.*δp, δ.*δm)
    end
end


function Base.size(ct::HelmholtzTensorLocalLinear)
    s = size(ct.elementgrid)
    st = length(ct.t)
    return (s[1], s[1], s[2], s[2], st)
end

function Base.getindex(ct::HelmholtzTensorLocalLinear, α::Int, β::Int, I::Int, J::Int, p::Int)
    r = abs(ct.elementgrid[α,I] - ct.elementgrid[β,J])
    δ = min(abs(ct.δp[α,I]-ct.δm[β,J]), abs(ct.δm[α,I]-ct.δp[β,J]))
    rmax = (r+δ)*ct.t[p]
    rmin = (r-δ)*ct.t[p]
    return ct.w[β,I]*0.5*√π*erf(rmin, rmax)/(rmax-rmin)
end

function Base.show(io::IO, ::MIME"text/plain", ct::HelmholtzTensorLocalLinear)
    s = size(ct)
    print(io, "HelmholtzTensorLocalLinear $(s[3]) elements and $(s[1]) Gauss points per element")
end


"""
    HelmholtzTensorLocalLog{T} <: AbstractHelmholtzTensorSingle

Helmholtz/Poisson equation tensor with logarithmic spacing for t-points and
t-integration local average correction.

If `k` is zero this tensor solves Poisson equation. If nonzero it works as
Helmholz equation Greens function.

# Fields
- `elementgrid::AbstractElementGrid{<:Any, 2}` : Gauss points and elements
- `t::Vector{Float64}`          : t-coordinates
- `wt::Vector{T}`               : Weights for t-integration
- `w::Matrix{Float64}`          : Weights for position coordinate integration
- `tmin::Float64`               : t-integration lower limit
- `tmax::Float64`               : t-integration upper limit
- `k::T`                        : Helmholz equation constant
- `δ::Float64`                  : Correction constant for t-integration ∈ ]0,1]
- `δp::Matrix{Float64}`         : Upper average points for t-integration correction
- `δm::Matrix{Float64}`         : Lower average points for t-integration correction

# Construction
    HelmholtzTensorLocalLog(eg::AbstractElementGridSymmetricBox, nt::Int; kwargs...)

## Keywords
- `tmin=0.`
- `tmax=25.`
- `δ=0.25`
- `k=0.`

# Example
```jldoctest
julia> ceg = CubicElementGrid(5u"Å", 4, 16)
Cubic elements grid with 4^3 elements with 16^3 Gauss points

julia> HelmholtzTensorLocalLog(ceg, 48)
HelmholtzTensorLocalLog 4 elements and 16 Gauss points per element

julia> ht = HelmholtzTensorLocalLog(ceg, 48; tmin=4., tmax=30., k=0.1)
HelmholtzTensorLocalLog 4 elements and 16 Gauss points per element

julia> typeof(ht) <: AbstractArray{Float64,5}
true

julia> ht[3,3,1,1,5] ≈ 0.10995313868
true
```
"""
struct HelmholtzTensorLocalLog{T} <: AbstractHelmholtzTensorSingle
    elementgrid::AbstractElementGrid{<:Any, 2}
    t::Vector{Float64}
    wt::Vector{T}
    w::Matrix{Float64}
    tmin::Float64
    tmax::Float64
    k::T
    δ::Float64
    δp::Matrix{Float64}
    δm::Matrix{Float64}
    function HelmholtzTensorLocalLog(eg::AbstractElementGridSymmetricBox, nt::Int;
                                   tmin=0., tmax=25., δ=0.25, k=0.)
        @assert 0 < δ <= 1
        @assert 0 <= tmin < tmax
        s, ws = gausspoints(nt; elementsize=(log(tmin+1e-12), log(tmax)))
        t = exp.(s)
        wt = ws .* t
        t, wt = gausspoints(nt; elementsize=(tmin, tmax))
        grid = get_1d_grid(eg)
        ss = size(grid)
        δp = zeros(ss...)
        δm = zeros(ss...)
        δm[2:end,:] = grid[1:end-1,:] .- grid[2:end,:]
        δp[1:end-1,:] = grid[2:end,:] .- grid[1:end-1,:]
        for j in 1:ss[2]
            δm[1,j] =  austrip( getelement(grid, j).low ) - grid[1,j]
            δp[end,j] = austrip( getelement(grid, j).high ) - grid[end,j]
        end
        wt = wt .* exp.(-(k^2)./(4t.^2))
        w = getweight(grid)
        new{eltype(wt)}(grid, t, wt, w, tmin, tmax, k, δ, δ.*δp, δ.*δm)
    end
end


function Base.size(ct::HelmholtzTensorLocalLog)
    s = size(ct.elementgrid)
    st = length(ct.t)
    return (s[1], s[1], s[2], s[2], st)
end

function Base.getindex(ht::HelmholtzTensorLocalLog, α::Int, β::Int, I::Int, J::Int, p::Int)
    r = abs(ht.elementgrid[α,I] - ht.elementgrid[β,J])
    δ = min(abs(ht.δp[α,I]-ht.δm[β,J]), abs(ht.δm[α,I]-ht.δp[β,J]))
    rmax = (r+δ)*ht.t[p]
    rmin = (r-δ)*ht.t[p]
    return ht.w[β,I]*0.5*√π*erf(rmin, rmax)/(rmax-rmin)
end

function Base.show(io::IO, ::MIME"text/plain", ht::HelmholtzTensorLocalLog)
    s = size(ht)
    print(io, "HelmholtzTensorLocalLog $(s[3]) elements and $(s[1]) Gauss points per element")
end


"""
    HelmholtzTensorCombination{T} <: AbstractHelmholtzTensor

Helmholtz/Poisson equation tensor that is composed of different types of tensors
with different t-integration ranges.

This tensor is intended to to be used to control how t-integration is performed.

# Fields
- `tensors::Vector{AbstractHelmholtzTensor}` : Helmholtz subtensors
- `ref::Vector{Int}`            : index of tensor wich is referenced at given t-point index
- `t::Vector{Float64}`          : t-integration points
- `wt::Vector{T}`               : weights for t-integration
- `tmin::Float64`               : t-integration lower limit
- `tmax::Float64`               : t-integration upper limit
- `tindex::Vector{Int}`         : Index of t in subtensor
- `k::T`                        : Helmholz equation constant

# Construction
    HelmholtzTensorCombination(t::HelmholtzTensorCombination...)


# Example
```jldoctest
julia> ceg = CubicElementGrid(5u"Å", 4, 16)
Cubic elements grid with 4^3 elements with 16^3 Gauss points

julia> ht1 = HelmholtzTensorLinear(ceg, 48)
HelmholtzTensorLinear 4 elements and 16 Gauss points per element

julia> ht2 = HelmholtzTensorLog(ceg, 48; tmin=25, tmax=50)
HelmholtzTensorLog 4 elements and 16 Gauss points per element

julia> hc = HelmholtzTensorCombination(ht1)
HelmholtzTensorCombination 4 elements and 16 Gauss points per element

julia> push!(hc, ht2)
HelmholtzTensorCombination 4 elements and 16 Gauss points per element

julia> HelmholtzTensorCombination(ht1, ht2)
HelmholtzTensorCombination 4 elements and 16 Gauss points per element

julia> typeof(hc) <: AbstractArray{Float64, 5}
true
```
"""
mutable struct HelmholtzTensorCombination{T} <: AbstractHelmholtzTensor
    tensors::Vector{AbstractHelmholtzTensor}
    ref::Vector{Int}
    t::Vector{Float64}
    wt::Vector{T}
    tmin::Float64
    tmax::Float64
    tindex::Vector{Int}
    k::T
    function HelmholtzTensorCombination(ht::AbstractHelmholtzTensor)
        s = size(ht)
        new{eltype(ht.wt)}([ht], ones(s[end]), ht.t, ht.wt, ht.tmin, ht.tmax, 1:s[end], ht.k)
    end
end


function HelmholtzTensorCombination(t::AbstractHelmholtzTensor...)
    ct = HelmholtzTensorCombination(t[1])
    for x in t[2:end]
        push!(ct,x)
    end
    return ct
end


function Base.size(ht::HelmholtzTensorCombination)
    s = size(ht.tensors[begin])
    return (s[1],s[2],s[3],s[4],length(ht.t))
end


function Base.getindex(ht::HelmholtzTensorCombination, i::Int, j::Int, I::Int, J::Int, ti::Int)
    return ht.tensors[ht.ref[ti]][i,j,I,J, ht.tindex[ti]]
end


function Base.show(io::IO, ::MIME"text/plain", ht::HelmholtzTensorCombination)
    s = size(ht)
    print(io, "HelmholtzTensorCombination $(s[3]) elements and $(s[1]) Gauss points per element")
end

function  Base.push!(htc::HelmholtzTensorCombination, tt::AbstractHelmholtzTensor) 
    @assert htc.tmax <= tt.tmin
    @assert htc.k == tt.k
    push!(htc.tensors,tt)
    htc.ref = vcat( htc.ref, ones( Int, length(tt.t)).*length(htc.tensors ) )
    htc.t = vcat( htc.t, tt.t)
    htc.wt = vcat( htc.wt, tt.wt )
    htc.tmax = tt.tmax
    htc.tindex = vcat( htc.tindex, 1:length(tt.t) )
    return htc
end

"""
    optimal_coulomb_tranformation(args...; kwargs...)

Returns Coulomb transformation tensor with optimal pre tested parameters.

# Args
- `ceg`                    :  grid information, either `CubicElementGrid` or something with `get_element_grid` implemented
- `nt::Int=96`             :  number of t-points

# Keywords
- `δ=0.25`          : parameter for local average correction
- `tmax=700`        : maximum t value
- `tboundary=20`    : point after which local average correction is applied
- `k=0.`            : constant for Helmholz equation
"""
function optimal_coulomb_tranformation(ceg::AbstractElementGrid, nt::Int=96; δ=0.25, tmax=700, tboundary=20, k=0.)
    @assert 0 < tboundary < tmax
    s, ws = gausspoints(nt; elementsize=(log(1e-12), log(tmax)))
    t = exp.(s)
    l = length(t[t .< tboundary])
    ct1 = HelmholtzTensorLog(ceg, l; tmax=tboundary, k=k)
    ct2 = HelmholtzTensorLocalLog(ceg, nt-l; tmin=tboundary, tmax=tmax, δ=δ, k=k)
    return HelmholtzTensorCombination(ct1,ct2)
end

function optimal_coulomb_tranformation(ceg, nt::Int=96; δ=0.25, tmax=700, tboundary=20, k=0.)
    tmp = get_elementgrid(ceg)
    return optimal_coulomb_tranformation( tmp, nt; δ=δ, tmax=tmax, tboundary=tboundary, k=k )
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



"""
    DerivativeTensor <: AbstractDerivativeTensor

Tensor that performs derivate over [`CubicElementGrid`](@ref).

# Fields
- `values::Matrix{Float64}`             : tensor values

# Construction
    DerivativeTensor(ceg::CubicElementGrid)
"""
struct DerivativeTensor <: AbstractDerivativeTensor
    values::Matrix{Float64}
    function DerivativeTensor(eg::AbstractElementGridSymmetricBox)
        new(get_derivative_matrix(eg))
    end
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
        # Derivative is done by analytically by deriving Gauss-Legendre basis
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
        new(ϕ)
    end
end


function Base.getindex(dt::DerivativeTensor, i::Int, j::Int)
    return dt.values[i,j]
end

Base.size(dt::DerivativeTensor) = size(dt.values)


function Base.show(io::IO, ::MIME"text/plain", dt::DerivativeTensor)
    print(io, "Derivative tensor size=$(size(dt))")
end

