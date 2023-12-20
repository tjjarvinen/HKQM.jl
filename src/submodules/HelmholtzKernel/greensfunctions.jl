abstract type AbstractTransformationTensor{T} <: AbstractArray{T,3} end

##

struct PoissonTensor{T} <: AbstractTransformationTensor{T}
    r_grid::AbstractElementGrid{T,1}
    t_grid::AbstractElementGrid{T,1}
    wr::Vector{T}
    wt::Vector{T}
    function PoissonTensor(
        r_grid::AbstractElementGrid{T, 1},
        t_grid::AbstractElementGrid{T, 1},
    ) where {T}
        wr = get_weight(r_grid)
        wt = get_weight(t_grid)
        new{T}(r_grid, t_grid, wr, wt)
    end
end


get_elementgrid(pt::PoissonTensor) = pt.r_grid

function Base.size(pt::PoissonTensor)
    a = size(pt.r_grid, 1)
    b = size(pt.t_grid, 1)
    return (a, a, b)
end

function Base.getindex(pt::PoissonTensor, i::Integer, j::Integer, ti::Integer)
    Δr = pt.r_grid[i] - pt.r_grid[j]
    t = pt.t_grid[ti]
    return exp( -(t*Δr)^2 ) * pt.wr[j] # j index is contracted
end

get_t_weight(pt::PoissonTensor, i::Integer) = pt.wt[i]

function Base.show(io::IO, ::MIME"text/plain", pt::PoissonTensor)
    s = size(pt)
    print(io, "Poisson tensor $(s[3]) t-points and $(s[1]) r-points")
end

get_t_max(pt::PoissonTensor) = element_bounds(pt.t_grid)[2]

function get_matrix_at_t(pt::PoissonTensor, t::Integer)
    return pt[:,:,t]    
end

##

struct HelmholtzTensor{T, Tt} <: AbstractTransformationTensor{T}
    tensor::AbstractTransformationTensor{T}
    k::Tt
    function HelmholtzTensor(pt::AbstractTransformationTensor{T}, k) where {T}
        new{T, typeof(k)}(pt, k)
    end
end

function HelmholtzTensor(
    r_grid::AbstractElementGrid{T, 1},
    t_grid::AbstractElementGrid{T, 1},
    k=0.0
) where {T}
    pt = PoissonTensor(r_grid, t_grid)
    return HelmholtzTensor(pt, k)    
end


Base.size(ht::HelmholtzTensor) = size(ht.tensor)

function Base.getindex(ht::HelmholtzTensor, i::Integer, j::Integer, ti::Integer)
    return ht.tensor[i, j, ti]
end

function get_t_weight(ht::HelmholtzTensor, i::Integer)
    t = ht.tensor.t_grid[i]
    return get_t_weight(ht.tensor, i) * exp( -(ht.k/(2t))^2 )
end

function Base.show(io::IO, ::MIME"text/plain", ht::HelmholtzTensor)
    s = size(ht)
    print(io, "Helmholtz tensor $(s[3]) t-points and $(s[1]) r-points")
end

get_t_max(ht::HelmholtzTensor) =get_t_max(ht.tensor)

function get_matrix_at_t(ht::HelmholtzTensor, t::Integer)
    return get_matrix_at_t(ht.tensor, t)   
end


##

function get_t_weight(t::AbstractTransformationTensor)
    return map( 1:size(t,3) ) do i
        get_t_weight(t, i)
    end
end

##

struct ConcreteTransformationTensor{T, TA} <: AbstractTransformationTensor{T}
    vals::TA
    tmax::T
    wt::Vector{T}
    function ConcreteTransformationTensor(tt::AbstractTransformationTensor{T}; array_type=Array) where {T}
        vals = array_type(tt)
        wt = [ get_t_weight(tt, i) for i in axes(tt,3) ]
        tmax = get_t_max(tt)
        new{T, typeof(vals)}(vals, tmax, wt)
    end
end

function Base.show(io::IO, ::MIME"text/plain", ct::ConcreteTransformationTensor)
    s = size(ct)
    print(io, "ConcreteTransformationTensor $(s[3]) t-points and $(s[1]) r-points. Array type is $(typeof(ct.vals))")
end

Base.size(ct::ConcreteTransformationTensor) = size(ct.vals)

Base.getindex(ct::ConcreteTransformationTensor, i...) = ct.vals[i...]

get_matrix_at_t(ct::ConcreteTransformationTensor, i) = view(ct.vals, :,:, i)

get_t_weight(ct::ConcreteTransformationTensor, i::Integer) = ct.wt[i]

get_t_max(ct::ConcreteTransformationTensor) = ct.tmax