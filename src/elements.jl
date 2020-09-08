
abstract type AbstractElementCenters end

"""
    ElementCenters <: AbstractElementCenters

Struct to hold finite element center locations

# Fields
- `centers::Array{Tuple{Float64,Float64,Float64},3` : stores centers

# Construct
    ElementCenters(n:Int; start=-10, stop=10)

# Arguments
- `n::Int` : number of boxes per dimension

# Keywords
- `start=-10` : minimum value for box center coordinate
- `stop=10`   : maximum value for box center coordinate

# Examples
```
julia> ElementCenters(10)
Finite element centers size=(10, 10, 10)
```
"""
struct ElementCenters <: AbstractElementCenters
    centers::Array{Tuple{Float64,Float64,Float64},3}
    function ElementCenters(n::Int; start=-10, stop=10)
        x = range(start, stop; length=n+2)[2:end-1]
        new(collect(Base.Iterators.product(x,x,x)))
    end
end

Base.size(ec::ElementCenters) = size(ec.centers)
Base.getindex(ec::ElementCenters, I...) = ec.centers[I...]

function Base.show(io::IO, ec::ElementCenters)
    print(io, "Finite element centers size=$(size(ec))")
end
