# Collection of all AbstractTypes

## Elements and Grids
abstract type AbstractElement{Dims} end
abstract type AbstractElementArray{T,N} <: AbstractArray{T,N} end
abstract type AbstractElementGrid{N} <: AbstractArray{SVector{3,Float64}, N} end

## Tensor types
abstract type AbtractTransformationTensor{T} <: AbstractArray{Float64, T} end
abstract type AbstractCoulombTransformation <: AbtractTransformationTensor{5} end
abstract type AbstractCoulombTransformationSingle{NT, NE, NG}  <: AbstractCoulombTransformation where {NT, NE, NG} end
abstract type AbstractCoulombTransformationCombination <: AbstractCoulombTransformation end
abstract type AbstractCoulombTransformationLocal{NT, NE, NG} <: AbstractCoulombTransformationSingle{NT, NE, NG} end

## State
abstract type AbstractQuantumState{T} <: AbstractArray{T,6} end


## Operators
abstract type AbstractOperator{N} end
abstract type AbstractScalarOperator <: AbstractOperator{1} end
abstract type AbstractVectorOperator <: AbstractOperator{3} end
abstract type AbstractCompositeOperator{N} <: AbstractOperator{N} end
abstract type AbstractHamiltonOperator <: AbstractOperator{1} end