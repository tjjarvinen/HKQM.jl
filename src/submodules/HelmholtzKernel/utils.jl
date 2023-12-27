

function default_transformation_tensor(r_grid::AbstractElementGrid{T, 1}; array_type=Array) where {T}
    tgrid = ElementGridVectorLegendre(ElementVector(T(0), T(20), T(100)), 32)
    return ConcreteTransformationTensor(PoissonTensor(r_grid, tgrid); array_type=array_type)
end

function default_transformation_tensor(r_grid::ElementGridArray, dimension::Integer; array_type=nothing)
    if isnothing(array_type)
        return default_transformation_tensor(r_grid.elements[dimension])
    else
        return default_transformation_tensor(r_grid.elements[dimension]; array_type=array_type)
    end
end


struct ElementGridArrayWithTransformation{T,TT,N} <: AbstractElementGrid{SVector{N,T}, N}
    ega::AbstractElementGrid{SVector{N,T}, N}
    transformations::Vector{TT}
    function ElementGridArrayWithTransformation(egv::AbstractElementGrid{T, 1}...; array_type=Array) where T
        ega = ElementGridArray(egv...)
        t = [ default_transformation_tensor( x ; array_type=array_type) for x in egv]
        new{T, eltype(t), length(egv)}(ega, t)
    end
end


function default_transformation_tensor(r_grid::ElementGridArrayWithTransformation, dimension::Integer; array_type=nothing)
    if isnothing(array_type)
        return r_grid.transformations[dimension]
    else
        return r_grid.transformations[dimension] |> array_type
    end
end


Base.size(egat::ElementGridArrayWithTransformation) = size(egat.ega)

Base.getindex(egat::ElementGridArrayWithTransformation, i...) = egat.ega[i...]

get_elementgrid(egat::ElementGridArrayWithTransformation, i::Integer) = get_elementgrid(egat.ega, i)
get_derivative_matrix(egat::ElementGridArrayWithTransformation, i::Integer) = get_derivative_matrix(egat.ega, i)
get_weight(egat::ElementGridArrayWithTransformation, i::Integer) = get_weight(egat.ega, i)

element_bounds(egat::ElementGridArrayWithTransformation, i::Integer) = element_bounds(egat.ega, i)
element_size(egat::ElementGridArrayWithTransformation, i::Integer) = element_size(egat.ega, i)


Base.similar(egat::ElementGridArrayWithTransformation) = similar(egat.ega)

Base.similar(egat::ElementGridArrayWithTransformation, T::Type{TO}) where {TO} = similar(egat.ega, T)

Base.fill(egat::ElementGridArrayWithTransformation, val) = fill(egat.ega, val)

Unitful.unit(egv::ElementGridArrayWithTransformation) = unit(egv.ega)