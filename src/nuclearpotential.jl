## Approximate PotentialTensor


"""
    nuclear_potential_harrison_approximation(args; kwargs)

Calculates Harrison style nuclear potential.
See [J. Chem. Phys. 121, 11587 (2004)](https://doi.org/10.1063/1.1791051)
for reference.

# Args
- `ceg`                       : Element grid used in calculation
- `r_atom::AbstractVector`    : Location of nucleus
- `atom_name::String`         : Atom symbol H, O, Na, etc.

# Kwargs
- `electron_charge=-1u"e_au"`   : Electron charge in further calculations
- `precision=1E-6`              : Desired precision
"""
function nuclear_potential_harrison_approximation(
        ceg,
        r_atom::AbstractVector,
        atom_name::String;
        electron_charge=-1u"e_au",
        precision=1E-6
    )
    @argcheck length(r_atom) == 3
    Z = elements[Symbol(atom_name)].number
    # Harrison's special term for scaling the potential
    # to desired accuracy
    c_param = cbrt( 0.00435 * precision / Z^5 )

    rr = position_operator(ceg) - r_atom
    # These two dont have unit
    r² = ( rr ⋅ rr ) / (c_param*u"bohr")^2 |> auconvert  
    r  = sqrt(r²) + 1E-10  # Make sure that no zero division

    U = erf(r)/r + 1/(3*√π) * ( exp(-r²) + 16exp(-4r²) )
    return 1u"hartree" / c_param * (Z * austrip(electron_charge) ) * U
end


## Tensors

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
    function NuclearPotentialTensor(r, ceg::AbstractElementGridSymmetricBox, nt::Int; tmin=0, tmax=30)
        @assert dimension(r) == NoDims || dimension(r) == dimension(u"m")
        rt = austrip(r)
        t, wt = gausspoints(nt; elementsize=(tmin, tmax))
        grid = get_1d_grid(ceg) .- rt 
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
    function NuclearPotentialTensorLog(r, ceg::AbstractElementGridSymmetricBox, nt::Int; tmin=0, tmax=30)
        @assert dimension(r) == NoDims || dimension(r) == dimension(u"m")
        rt = austrip(r)
        s, ws = gausspoints(nt; elementsize=(log(tmin+1e-12), log(tmax)))
        t = exp.(s)
        wt = ws .* t
        grid = get_1d_grid(ceg) .- rt
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
    function NuclearPotentialTensorLogLocal(r, ceg::AbstractElementGridSymmetricBox, nt::Int;
                                           tmin=0, tmax=30, δ=0.25)
        @assert 0 < δ <= 1
        @assert 0 <= tmin < tmax
        @assert dimension(r) == NoDims || dimension(r) == dimension(u"m")
        rt = austrip(r)
        s, ws = gausspoints(nt; elementsize=(log(tmin+1e-12), log(tmax)))
        t = exp.(s)
        wt = ws .* t
        t, wt = gausspoints(nt; elementsize=(tmin, tmax))
        tmp = get_1d_grid(ceg)
        grid = tmp .- rt
        ss = size(grid)

        # get range around points [x+δm, x+δp]
        # and calculate average on that range (later on)
        δp = zeros(ss...)
        δm = zeros(ss...)
        δm[2:end,:] = grid[1:end-1,:] .- grid[2:end,:]
        δp[1:end-1,:] = grid[2:end,:] .- grid[1:end-1,:]
        for j in 1:ss[2]
            δm[1,j] =  austrip( getelement(tmp, j).low ) - grid[1,j]
            δp[end,j] = austrip( getelement(tmp, j).high ) - grid[end,j]
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
    NuclearPotentialTensorGaussian(r, ceg::CubicElementGrid, nt, β=100; tmin=0, tmax=20)

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
    function NuclearPotentialTensorGaussian(r, ceg::AbstractElementGridSymmetricBox, nt, β; tmin=0, tmax=20)
        @assert dimension(r) == NoDims || dimension(r) == dimension(u"m")
        rt = austrip(r)
        s, ws = gausspoints(nt; elementsize=(log(tmin+1e-12), log(tmax)))
        t = exp.(s)
        wt = ws .* t
        grid = get_1d_grid(ceg) .- rt
        new{eltype(grid)}(grid, t, wt, tmin, tmax, rt, β)
    end
end

function NuclearPotentialTensorGaussian(r, ceg::AbstractElementGridSymmetricBox, nt; σ=0.1, tmin=0, tmax=20)
    egv = get_1d_grid(ceg)
    min_d = element_size(egv.elements[1].element) / size(ceg)[1] |> austrip
    #min_d = elementsize(ceg.elements) / size(ceg)[1] |> austrip
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

# function Base.Array(pt::PotentialTensor)
#     x = Array(pt.x)
#     y = Array(pt.y)
#     z = Array(pt.z)
#     @tullio out[i,j,k,I,J,K] := x[i,I,t] * y[j,J,t] * z[k,K,t] * pt.x.wt[t]
#     return 2/sqrt(π) .* out
# end


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



## New

struct NuclearPotentialTensorIntegral{T} <: AbstractNuclearPotentialSingle{T}
    elementgrid::ElementGridVector
    t::Vector{Float64}
    wt::Vector{Float64}
    tmin::Float64
    tmax::Float64
    r::T
    function NuclearPotentialTensorIntegral(r, ceg::AbstractElementGridSymmetricBox, nt; tmin=0., tmax=20. )
        @assert dimension(r) == NoDims || dimension(r) == dimension(u"m")
        rt = austrip(r)
        t, wt = gausspoints(nt; elementsize=(tmin, tmax))
        grid = get_1d_grid(ceg)
        new{typeof(rt+1.0)}(grid, t, wt, tmin, tmax, rt)
    end
end

Base.show(io::IO, ::MIME"text/plain", ::NuclearPotentialTensorIntegral) = print(io, "NuclearPotentialTensorIntegral")

function Base.size(npti::NuclearPotentialTensorIntegral)
    s = size(npti.elementgrid)
    return (s[1], s[2], length(npti.t))
end


function Base.getindex(npti::NuclearPotentialTensorIntegral, i::Int, I::Int, t::Int)
    _f(x, u) = npti.elementgrid.elements[I](x, u)   # Function to interpolate Gauss-Lagrange polynomials
    l = npti.elementgrid.elements[I].element.low |> austrip
    h = npti.elementgrid.elements[I].element.high |> austrip

    u = [ k==i ? 1. : 0. for k in axes(npti,1) ] # Interpolation weights
    t² = npti.t[t]^2
    integral, err = quadgk( r -> exp(-t²*(r-npti.r)^2 ) * _f(r, u), l, h; rtol=1e8)
    return integral
end


struct NuclearPotentialTensorLogIntegral{T} <: AbstractNuclearPotentialSingle{T}
    elementgrid::ElementGridVector
    t::Vector{Float64}
    wt::Vector{Float64}
    tmin::Float64
    tmax::Float64
    r::T
    function NuclearPotentialTensorLogIntegral(r, ceg::AbstractElementGridSymmetricBox, nt; tmin=0., tmax=20. )
        @assert dimension(r) == NoDims || dimension(r) == dimension(u"m")
        rt = austrip(r)
        s, ws = gausspoints(nt; elementsize=(log(tmin+1e-12), log(tmax)))
        t = exp.(s)
        wt = ws .* t
        grid = get_1d_grid(ceg)
        new{typeof(rt+1.0)}(grid, t, wt, tmin, tmax, rt)
    end
end


Base.show(io::IO, ::MIME"text/plain", ::NuclearPotentialTensorLogIntegral) = print(io, "NuclearPotentialTensorIntegral")

function Base.size(npti::NuclearPotentialTensorLogIntegral)
    s = size(npti.elementgrid)
    return (s[1], s[2], length(npti.t))
end

function Base.getindex(npti::NuclearPotentialTensorLogIntegral, i::Int, I::Int, t::Int)
    _f(x, u) = npti.elementgrid.elements[I](x, u)   # Function to interpolate Gauss-Lagrange polynomials
    l = npti.elementgrid.elements[I].element.low |> austrip
    h = npti.elementgrid.elements[I].element.high |> austrip

    u = [ k==i ? 1. : 0. for k in axes(npti,1) ] # Interpolation weights
    t² = npti.t[t]^2

    integral, err = quadgk( r -> exp(-t²*(r-npti.r)^2 ) * _f(r, u), l, h; rtol=1e8)
    return integral
end
