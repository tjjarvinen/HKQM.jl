



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
    min_d = elementsize(ceg.elements) / size(ceg)[1] |> austrip
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
