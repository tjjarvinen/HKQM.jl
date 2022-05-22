# General operator stuff


get_elementgrid(ao::AbstractOperator) = ao.elementgrid

Base.size(ao::AbstractOperator) = size(get_elementgrid(ao))

function Base.show(io::IO, ao::AbstractOperator)
    s = size(ao)
    print(io, "Operator $(s[end])^3 elements, $(s[1])^3 Gauss points per element")
end

Base.length(::AbstractOperator{N}) where N = N
Base.firstindex(::AbstractOperator) = 1
Base.lastindex(ao::AbstractOperator) = length(ao)


function Base.getindex(so::AbstractOperator{1}, i::Int)
    @assert i == 1
    return so
end

function Base.iterate(v::AbstractOperator, i=0)
    if i < length(v)
        return (v[i+1], i+1)
    else
        return nothing
    end
end

Unitful.unit(ao::AbstractOperator) = ao.unit
Unitful.dimension(ao::AbstractOperator) = dimension(unit(ao))

function Unitful.uconvert(u::Unitful.Units, op::AbstractOperator)
    @assert dimension(u) == dimension(op)
    unit(op) == u && return op
    conv = ustrip(uconvert(u, 1*unit(op))) * u / unit(op)
    return conv*op
end

Base.:(+)(ao::AbstractOperator) = ao
Base.:(-)(ao::AbstractOperator) = -1*ao

dot(v1::AbstractOperator{N}, v2::AbstractOperator{N}) where N = sum(v1.*v2)

function cross(v1::AbstractOperator{3}, v2::AbstractOperator{3})
    r1 = v1[2]*v2[3] - v1[3]*v2[2]
    r2 = v1[1]*v2[3] - v1[3]*v2[1]
    r3 = v1[1]*v2[2] - v1[2]*v2[1]
    return VectorOperator(r1,r2,r3)
end

Base.:(*)(op::AbstractOperator{1}, qs::QuantumState) = op(qs)
Base.:(*)(op::AbstractOperator, qs::QuantumState) = map(f->f(qs), op)

## Scalar Operator

"""
    ScalarOperator{T}

Type for operators that are simple multiplications. This means that the
operator is simply an array.

# Fields
- `elementgrid::CubicElementGrid`  :  grid information
- `vals::T`                        :  operator values
- `unit::Unitful.FreeUnits`        :  unit for operator

# Examples
```jldoctest
julia> ceg = CubicElementGrid(5u"Å", 2, 32)
Cubic elements grid with 2^3 elements with 32^3 Gauss points

julia> op = ScalarOperator(ceg, ones(size(ceg)))
Operator 2^3 elements, 32^3 Gauss points per element

julia> sin_op = sin(op)
Operator 2^3 elements, 32^3 Gauss points per element

julia> sin_op + 3op
Operator 2^3 elements, 32^3 Gauss points per element

julia> ψ = QuantumState(ceg, ones(size(ceg)))
Quantum state

julia> -2op * ψ
Quantum state
```
"""
struct ScalarOperator{T} <: AbstractScalarOperator
    elementgrid::AbstractElementGrid{SVector{3,Float64},6}
    vals::T
    unit::Unitful.FreeUnits
    function ScalarOperator(ceg::AbstractElementGrid{SVector{3,Float64},6},
                            ψ::AbstractArray{<:Any,6}
                            ;unit::Unitful.FreeUnits=NoUnits)
        new{typeof(ψ)}(ceg, ψ, unit)
    end
end

get_values(so::ScalarOperator) = so.vals

(so::ScalarOperator)(ψ::AbstractArray{<:Any,6}) = QuantumState(so.elementgrid, so.vals.*ψ, unit(so))
function (so::ScalarOperator)(qs::QuantumState)
    @assert size(so) == size(qs)
    return QuantumState(so.elementgrid, so.vals.*qs.psi, unit(so)*unit(qs))
end

for OP in (:(+), :(-))
    @eval begin
        function Base.$OP(a::ScalarOperator,b::ScalarOperator)
            @assert dimension(a) == dimension(b) "Dimension missmatch"
            c = uconvert(unit(a), b)
            ScalarOperator(a.elementgrid, $OP.(a.vals,c.vals); unit=a.unit)
        end
        function Base.$OP(a::ScalarOperator, x::Number)
            @assert dimension(a) == dimension(x) "Dimension missmatch"
            y = ustrip(uconvert(unit(a), x))
            return ScalarOperator(a.elementgrid, $OP.(a.vals,y); unit=a.unit)
        end
        function Base.$OP(x::Number, a::ScalarOperator)
            @assert dimension(a) == dimension(x) "Dimension missmatch"
            y = ustrip(uconvert(unit(a), x))
            return ScalarOperator(a.elementgrid, $OP.(y,a.vals); unit=a.unit)
        end
    end
end

for OP in (:(/), :(*))
    @eval begin
        function Base.$OP(a::ScalarOperator,b::ScalarOperator)
            ScalarOperator(a.elementgrid, $OP.(a.vals,b.vals); unit=$OP(unit(a), unit(b)))
        end
        function Base.$OP(a::ScalarOperator, x::Number)
            return ScalarOperator(a.elementgrid, $OP.(a.vals,ustrip(x)); unit=$OP(unit(a), unit(x)))
        end
        function Base.$OP(x::Number, a::ScalarOperator)
            return ScalarOperator(a.elementgrid, $OP.(ustrip(x),a.vals); unit=$OP(unit(x), unit(a)))
        end
    end
end

Base.:(-)(a::ScalarOperator) = -1*a
Base.:(^)(a::ScalarOperator, x::Number) = ScalarOperator(a.elementgrid, (^).(a.vals,x); unit=unit(a)^x)


for op in (:sqrt, :cbrt, :real, :imag)
    @eval Base.$op(so::ScalarOperator) = ScalarOperator(so.elementgrid, $op.(so.vals); unit=$op(so.unit))
end

for op in (:sin, :cos, :tan, :exp, :log)
    @eval function Base.$op(so::ScalarOperator)
        @assert dimension(so) == dimension(NoUnits) "Operator needs to be dimensionless"
        so1 = auconvert(so)
        return ScalarOperator(so1.elementgrid, $op.(so1.vals))
     end
end


for op in (:erf,)
    #NOTE does not work with GPUs
    @eval function SpecialFunctions.$op(so::ScalarOperator)
        @assert dimension(so) == dimension(NoUnits) "Operator needs to be dimensionless"
        so1 = auconvert(so)
        b = similar(so.vals)
        Threads.@threads for i in eachindex(b)
            #@inbounds b[i] = $op(so1.vals[i])
            b[i] = $op(so1.vals[i])
        end
        return ScalarOperator(get_elementgrid(so), b)
     end
end


function Unitful.uconvert(a::Unitful.Units, so::ScalarOperator)
    @assert dimension(a) == dimension(so) "Dimension missmatch"
    a == unit(so) && return so
    conv = ustrip(uconvert(a, 1*unit(so)))
    return ScalarOperator(so.elementgrid, so.vals .* conv; unit=a)
end


## Vector Operator

"""
    VectorOperator{N}

Operetor that can be considered to be a vector. This type is build out of several
scalar operators.

# Fields
- `elementgrid::CubicElementGrid`  : grid information
- `operators::Vector{AbstractOperator{1}}`    : scalar operator from which vector
    is build

# Construction
    VectorOperator(op::AbstractOperator{1}...)
"""
struct VectorOperator{N} <: AbstractOperator{N}
    elementgrid::AbstractElementGrid{SVector{3,Float64},6}
    operators::Vector{AbstractOperator{1}}
    function VectorOperator(op::AbstractOperator{1}...)
        si = size.(op)
        @assert all(map(x-> x==si[begin], si)) "Scalar operators have different sizes"
        new{length(op)}(op[1].elementgrid, [op...])
    end
    function VectorOperator(op::AbstractVector{<:AbstractOperator{1}})
        si = size.(op)
        @assert all(map(x-> x==si[begin], si)) "Scalar operators have different sizes"
        new{length(op)}(op[1].elementgrid, op)
    end
end

Base.firstindex(::VectorOperator) = 1
Base.getindex(v::VectorOperator, i...) = v.operators[i...]
Base.lastindex(v::VectorOperator) = length(v)

Unitful.unit(v::VectorOperator) = unit(v[begin])

for OP in (:(+), :(-))
    @eval begin
        function Base.$OP(a::VectorOperator{N},b::VectorOperator{N}) where N
            VectorOperator($OP(a.operators,b.operators))
        end
        function Base.$OP(a::VectorOperator{N},b::AbstractVector) where N
            @assert length(b) == N
            VectorOperator($OP(a.operators,b))
        end
        function Base.$OP(a::AbstractVector,b::VectorOperator{N}) where N
            @assert length(a) == N
            VectorOperator($OP(a,b.operators))
        end
    end
end

for OP in (:(+), :(-), :(*))
    @eval begin
        Base.$OP(a::VectorOperator, x::Number) = VectorOperator($OP.(a.operators,x))
        Base.$OP(x::Number, a::VectorOperator) = VectorOperator($OP.(x,a.operators))

        function Base.$OP(v::VectorOperator,s::ScalarOperator)
            @assert size(v) == size(s)
            VectorOperator( [$OP(x,s) for x in v.operators] )
        end
        function Base.$OP(s::ScalarOperator,v::VectorOperator)
            @assert size(v) == size(s)
            VectorOperator( [$OP(s,x) for x in v.operators] )
        end
    end
end


for OP in [:(/)]
    @eval begin
        function Base.$OP(v::VectorOperator,s::ScalarOperator)
            @assert size(v) == size(s)
            VectorOperator( [$OP(x,s) for x in v.operators] )
        end
        Base.$OP(a::VectorOperator, x::Number) = VectorOperator($OP.(a.operators,x))
    end
end


Base.:(-)(a::VectorOperator) = VectorOperator( [-x for x in a]  )


## Position operator

"""
    position_operator(ceg::AbstractElementGrid{SVector{3,Float64}, 6})

Returns position operator as a [`VectorOperator`](@ref).
"""
function position_operator(ceg::AbstractElementGrid{SVector{3,Float64}, 6})
    x(v) = v[1]
    y(v) = v[2]
    z(v) = v[3]
    return VectorOperator(
                          ScalarOperator(ceg, x.(ceg); unit=u"bohr"),
                          ScalarOperator(ceg, y.(ceg); unit=u"bohr"),
                          ScalarOperator(ceg, z.(ceg); unit=u"bohr")
                         )
end


## Derivative Operator

"""
    DerivativeOperator{NG,N}

Calculates derivative when operating. `N` is coordinate that is derived.

# Fields
- `elementgrid::CubicElementGrid`   :  grid information
- `dt::DerivativeTensor{NG}`        :  derivative in tensor form

# Creation
    DerivativeOperator(ceg::CubicElementGrid, coordinatate::Int=1)
    DerivativeOperator(ceg::CubicElementGrid, dt::DerivativeTensor, coordinatate::Int=1)
"""
struct DerivativeOperator{N} <: AbstractOperator{1}
    elementgrid::AbstractElementGrid{SVector{3,Float64},6}
    dt::DerivativeTensor
    function DerivativeOperator(ceg::AbstractElementGrid{SVector{3,Float64},6}, coordinatate::Int=1)
        @assert coordinatate ∈ 1:3 "coordinate needs to be 1, 2 or 3"
        dt = DerivativeTensor(ceg)
        new{coordinatate}(ceg, dt)
    end
    function DerivativeOperator(ceg::AbstractElementGrid{SVector{3,Float64},6}, dt::DerivativeTensor, coordinatate::Int=1)
        @assert coordinatate ∈ 1:3 "coordinate needs to be 1, 2 or 3"
        @assert size(ceg)[1] == size(dt)[1]
        new{coordinatate}(ceg, dt)
    end
end


function (d::DerivativeOperator{1})(qs::QuantumState)
    return QuantumState(d.elementgrid, operate_x(d.dt, qs.psi), unit(qs)*unit(d) )
end

function (d::DerivativeOperator{2})(qs::QuantumState)
    return QuantumState(d.elementgrid, operate_y(d.dt, qs.psi), unit(qs)*unit(d) )
end

function (d::DerivativeOperator{3})(qs::QuantumState)
    return QuantumState(d.elementgrid, operate_z(d.dt, qs.psi), unit(qs)*unit(d) )
end


Unitful.unit(::DerivativeOperator) = u"bohr"^-1

## Gradient Operator

"""
    GradientOperator{NG}

Calculates gradient when operating.

# Fields
- `elementgrid::CubicElementGrid`   :  grid information
- `dt::DerivativeTensor{NG}`        :  derivative in tensor form

# Creation
    GradientOperator(ceg::CubicElementGrid)
"""
struct GradientOperator <: AbstractOperator{3}
    elementgrid::AbstractElementGridSymmetricBox
    dt::DerivativeTensor
    function GradientOperator(ceg::AbstractElementGridSymmetricBox)
        dt = DerivativeTensor(ceg)
        new(ceg, dt)
    end
end


function (go::GradientOperator)(ψ::QuantumState)
    return map(f->f(ψ), go)
end

function (go::GradientOperator)(a::AbstractVector{<:AbstractQuantumState})
    @assert length(a) == 3
    @assert all(map(x->size(x)==size(go), a))
    out = operate_x(go.dt, a[1].ψ)
    out .+= operate_y(go.dt, a[2].ψ)
    out .+= operate_z(go.dt, a[3].ψ)
    return QuantumState(go.ceg, out, unit(a)*unit(g))
end

#(go::GradientOperator)(ψ::QuantumState) = go(ψ.ψ)
Base.length(::GradientOperator) = 3
Base.show(io::IO, go::GradientOperator) = print(io::IO, "Gradient operator size=$(size(go))")

function Base.getindex(go::GradientOperator, i::Int)
    @assert 1<=i<=3
    return DerivativeOperator(go.elementgrid, go.dt, i)
end

Unitful.unit(::GradientOperator) = u"bohr"^-1

## derivative operations
# these are low level stuff to use only in special cases

function operate_x!(dψ::AbstractArray{<:Any,6}, dt::DerivativeTensor, ψ::AbstractArray{<:Any,6})
    tmp = similar(ψ, size(dt)...)
    tmp .= dt
    @tensor dψ[i,j,k,I,J,K] = tmp[i,l] * ψ[l,j,k,I,J,K]
    return dψ
end

function operate_y!(dψ::AbstractArray{<:Any,6}, dt::DerivativeTensor, ψ::AbstractArray{<:Any,6})
    tmp = similar(ψ, size(dt)...)
    tmp .= dt
    @tensor dψ[i,j,k,I,J,K] = tmp[j,l] * ψ[i,l,k,I,J,K]
    return dψ
end

function operate_z!(dψ::AbstractArray{<:Any,6}, dt::DerivativeTensor, ψ::AbstractArray{<:Any,6})
    tmp = similar(ψ, size(dt)...)
    tmp .= dt
    @tensor dψ[i,j,k,I,J,K] = tmp[k,l] * ψ[i,j,l,I,J,K]
    return dψ
end


function operate_x(dt::DerivativeTensor, ψ::AbstractArray{<:Any,6})
    tmp = similar(ψ)
    return operate_x!(tmp, dt, ψ)
end

function operate_y(dt::DerivativeTensor, ψ::AbstractArray{<:Any,6})
    tmp = similar(ψ)
    return operate_y!(tmp, dt, ψ)
end

function operate_z(dt::DerivativeTensor, ψ::AbstractArray{<:Any,6})
    tmp = similar(ψ)
    return operate_z!(tmp, dt, ψ)
end

## Constant operator

"""
    ConstantTimesOperator{TO,T,N}

Type to represent constant times operator. It is recommended not to create this
directly. This type is created when there is no simple way to implement operator
times constant operation.

# Fields
- `elementgrid::AbstractElementGrid`  : grid information
- `op::TO`                         : operator
- `a::T`                           : constant
"""
struct ConstantTimesOperator{TO,T,N} <: AbstractOperator{N}
    elementgrid::AbstractElementGrid
    op::TO
    a::T
    function ConstantTimesOperator(op::AbstractOperator{N}, a=1) where N
        if ustrip(imag(a)) ≈ 0
            new{typeof(op),typeof(real(a)), N}(get_elementgrid(op), op, real(a))
        else
            new{typeof(op),typeof(a), N}(get_elementgrid(op), op, a)
        end
    end
end

Base.:(*)(op::AbstractOperator, a::Number) = ConstantTimesOperator(op,a)
Base.:(*)(a::Number, op::AbstractOperator) = ConstantTimesOperator(op,a)
Base.:(/)(op::AbstractOperator, a::Number) = ConstantTimesOperator(op,1/a)

Base.getindex(cto::ConstantTimesOperator,i::Int) = cto.a*cto.op[i]

Unitful.unit(op::ConstantTimesOperator) = unit(op.op) * unit(op.a)

function (cto::ConstantTimesOperator{TO,T,1})(qs::QuantumState) where {TO,T}
    return cto.a * cto.op(qs)
end

function dot(cto1::ConstantTimesOperator{TO,T,N}, cto2::ConstantTimesOperator{TO,T,N}) where {TO,T,N}
    return ConstantTimesOperator(dot(cto1.op,cto2.op), cto1.a*cto2.a)
end

function dot(cto::ConstantTimesOperator{TO,T,N}, op::AbstractOperator{N}) where {TO,T,N}
    return ConstantTimesOperator(dot(cto.op,op), cto.a)
end

function dot(op::AbstractOperator{N}, cto::ConstantTimesOperator{TO,T,N}) where {TO,T,N}
    return ConstantTimesOperator(dot(cto.op,op), cto.a)
end




## Operator sum

"""
    OperatorSum{TO1, TO2, N}

Sum of two operator that cannot directly summed. It is recommended not to create
this directly.

# Fields
- `elementgrid::AbstractElementGrid`   :  grid inforamation
- `op1::TO1`                        :  operator 1
- `op2::TO2`                        :  operator 2
"""
struct OperatorSum{TO1, TO2, N} <: AbstractCompositeOperator{N}
    elementgrid::AbstractElementGrid
    op1::TO1
    op2::TO2
    function OperatorSum(op1::AbstractOperator{N}, op2::AbstractOperator{N}) where N
        @assert size(op1) == size(op2)
        @assert dimension(op1) == dimension(op2) "Operators have different unit dimensions"
        if unit(op1) == unit(op2)
            new{typeof(op1),typeof(op2),N}(get_elementgrid(op1), op1, op2)
        else
            new{typeof(op1),typeof(op2),N}(get_elementgrid(op1), op1, uconvert(unit(op1), op2))
        end
    end
end

function Base.:(+)(op1::AbstractOperator{N}, op2::AbstractOperator{N}) where N
    return OperatorSum(op1,op2)
end

function Base.:(-)(op1::AbstractOperator{N}, op2::AbstractOperator{N}) where N
    return OperatorSum(op1,-op2)
end

function (os::OperatorSum{TO1,TO2, 1})(qs::QuantumState) where {TO1,TO2}
    return os.op1(qs) + os.op2(qs)
end

Unitful.unit(sos::OperatorSum) = unit(sos.op1)

Base.getindex(op::OperatorSum, i::Int) = op.op1[i] + op.op2[i]

## Scalar operator product

"""
    OperatorProduct{TO1, TO2}

Producto of two operators that cannot be them selves multiplied. These to operator
are operated after each other. Operator 2 is operated first and then operator 1.
It is recommended not to create this directly.

# Fields
- `elementgrid::AbstractElementGrid`   :  grid information
- `op1::TO1`                        :  operator 1
- `op2::TO2`                        :  operator 2
"""
struct OperatorProduct{TO1, TO2} <: AbstractCompositeOperator{1}
    elementgrid::AbstractElementGrid
    op1::TO1
    op2::TO2
    function OperatorProduct(op1::AbstractOperator{1}, op2::AbstractOperator{1})
        @assert size(op1) == size(op2)
        @assert length(op1) == length(op2)  "Operators have different length"
        new{typeof(op1),typeof(op2)}(get_elementgrid(op1), op1, op2)
    end
end

function Base.:(*)(op1::AbstractOperator{1}, op2::AbstractOperator{1})
    return OperatorProduct(op1,op2)
end

(op::OperatorProduct)(qs::QuantumState) = op.op1(op.op2(qs))

Unitful.unit(op::OperatorProduct) = unit(op.op1) * unit(op.op2)

get_elementgrid(op::OperatorProduct) = get_elementgrid(op.op1)


## Laplace Operator

"""
    LaplaceOperator

∇² operator

# Fields
- `elementgrid::AbstractElementGrid`   :  grid information
- `g::GradientOperator`             :  gradient operator

# Creation
    LaplaceOperator(g::GradientOperator)
"""
struct LaplaceOperator <: AbstractOperator{1}
    elementgrid::AbstractElementGrid
    g::GradientOperator
    function LaplaceOperator(g::GradientOperator)
        new(get_elementgrid(g), g)
    end
    function LaplaceOperator(ceg::AbstractElementGrid)
        new(ceg, GradientOperator(ceg))
    end
end

Unitful.unit(lo::LaplaceOperator) = unit(lo.g)^2

Base.:(*)(g1::GradientOperator, g2::GradientOperator) = LaplaceOperator(g1)

dot(g1::GradientOperator, g2::GradientOperator) = LaplaceOperator(g1)

function (lo::LaplaceOperator)(qs::QuantumState)
    tmp = similar(qs.psi, size(lo.g.dt))
    tmp .= lo.g.dt
    @tensor w[i,j]:=tmp[i,k]*tmp[k,j]
    @tensor ϕ[i,j,k,I,J,K]:= w[i,ii]*qs.psi[ii,j,k,I,J,K] + w[j,jj]*qs.psi[i,jj,k,I,J,K] + w[k,kk]*qs.psi[i,j,kk,I,J,K]
    return QuantumState(get_elementgrid(qs), ϕ, unit(qs)*unit(lo.g)^2)
end


## Special operators

"""
    momentum_operator(ceg::AbstractElementGrid)
    momentum_operator(qs::QuantumState)
    momentum_operator(H::HamiltonOperator)
    momentum_operator(H::HamiltonOperatorFreeParticle)
    momentum_operator(H::HamiltonOperatorMagneticField)

Return momentum operator. For Hamiltonian operator input this returns
canonical momentum.
"""
function momentum_operator(ceg::AbstractElementGrid)
    c = -im * 1u"ħ_au"
    g = GradientOperator(ceg)
    return c*g
end

momentum_operator(qs::QuantumState) = momentum_operator(qs.elementgrid)

function kinetic_energy_operator(ceg::AbstractElementGrid, m=1u"me_au")
    @assert dimension(m) == dimension(u"kg")
    return HamiltonOperatorFreeParticle(ceg; m=m)
end

## Hamilton operator

"""
    HamiltonOperatorFreeParticle{T}

Free particle Hamiltonian

# Fields
- `elementgrid::AbstractElementGrid`   :  grid information
- `∇²::LaplaceOperator`             :  Laplace operator
- `m::T`                            :  mass of the particle

# Construction
    HamiltonOperatorFreeParticle(ceg::AbstractElementGrid;m=1u"me_au")
"""
struct HamiltonOperatorFreeParticle{T} <: AbstractHamiltonOperator
    elementgrid::AbstractElementGrid
    ∇²::LaplaceOperator
    m::T
    function HamiltonOperatorFreeParticle(ceg::AbstractElementGrid;m=1u"me_au")
        @assert dimension(m) == dimension(u"kg")
        new{typeof(m)}(ceg, LaplaceOperator(ceg), m)
    end
end

Unitful.unit(H::HamiltonOperatorFreeParticle) =u"hartree*bohr^2"*unit(H.∇²)

function (H::HamiltonOperatorFreeParticle)(qs::QuantumState)
    return uconvert(u"hartree*bohr^2", u"ħ"^2/(-2H.m))*(H.∇²*qs)
end

"""
    HamiltonOperator{TF,TV}

Hamiltonian without magnetic field.

# Fields
- `elementgrid::AbstractElementGrid`         :  grid information
- `T::HamiltonOperatorFreeParticle{TF}`   :  kinetic energy operator
- `V::TV`                                 :  potential energy operator

# Creation
    HamiltonOperator(V::AbstractOperator{1}; m=1u"me_au")
"""
struct HamiltonOperator{TF,TV} <: AbstractHamiltonOperator
    elementgrid::AbstractElementGrid
    T::HamiltonOperatorFreeParticle{TF}
    V::TV
    function HamiltonOperator(V::AbstractOperator{1}; m=1u"me_au")
        @assert dimension(V) == dimension(u"J")
        @assert dimension(m) == dimension(u"kg")
        T = HamiltonOperatorFreeParticle(get_elementgrid(V); m=m)
        new{typeof(T.m),typeof(V)}(get_elementgrid(V), T, V)
    end
end

Unitful.unit(H::HamiltonOperator) = unit(H.T)

(H::HamiltonOperator)(qs::QuantumState) = H.T*qs + H.V*qs


"""
    HamiltonOperatorMagneticField{TF,TV,TA,Tq}

Hamiltonian with magnetic field.

# Fields
- `elementgrid::AbstractElementGrid`         :  grid information
- `T::HamiltonOperatorFreeParticle{TF}`   :  kinetic energy operator
- `V::TV`                                 :  potential enegy operator
- `A::TA`                                 :  vector potential
- `q::Tq`                                 :  electric charge

# Construction
    HamiltonOperatorMagneticField(Args...; Kwargs...)

## Arguments for construction
- `V::AbstractOperator{1}`    :  potential energy
- `A::AbstractOperator{3}`    :  vector potential

## Keywords for construction
- `m=1u"me_au"`               :  mass of the particle
- `q=-1u"e_au"`               :  charge for the particle
"""
struct HamiltonOperatorMagneticField{TF,TV,TA,Tq} <: AbstractHamiltonOperator
    elementgrid::AbstractElementGrid
    T::HamiltonOperatorFreeParticle{TF}
    V::TV
    A::TA
    q::Tq
    function HamiltonOperatorMagneticField(
            V::AbstractOperator{1},
            A::AbstractOperator{3};
            m=1u"me_au",
            q=-1u"e_au"
        )
        @assert dimension(V) == dimension(u"J")
        @assert dimension(A) == dimension(u"T*m")
        @assert dimension(m) == dimension(u"kg")
        @assert dimension(q) == dimension(u"e_au")
        T = HamiltonOperatorFreeParticle(V.elementgrid; m=m)
        new{typeof(T.m),typeof(V),typeof(A),typeof(q)}(get_elementgrid(V), T, V, A, q)
    end
end

Unitful.unit(H::HamiltonOperatorMagneticField) = unit(H.T)

function (H::HamiltonOperatorMagneticField)(ψ::QuantumState)
    #TODO make this better
    p = momentum_operator(H.T)
    ϕ = H.q^2*(H.A⋅H.A)*ψ
    ϕ += H.q*(p⋅H.A)*ψ
    ϕ += H.q*(H.A⋅p)*ψ
    ϕ /= 2H.T.m
    ϕ += H.V*ψ
    return H.T*ψ + ϕ
end

"""
    vector_potential(ceg, Bx, By, Bz)

Gives vector potential for constant magnetic field.
"""
function vector_potential(ceg, Bx, By, Bz)
    @assert dimension(Bx) == dimension(By) == dimension(Bz) == dimension(u"T")
    r = position_operator(ceg)
    # A = r×B/2
    Ax = 0.5*(r[2]*Bz-r[3]*By)
    Ay = 0.5*(r[3]*Bx-r[1]*Bz)
    Az = 0.5*(r[1]*By-r[2]*Bx)
    return VectorOperator(Ax, Ay, Az)
end

function momentum_operator(H::HamiltonOperatorMagneticField)
    p = momentum_operator(H.elementgrid)
    return p + H.q*H.A
end

momentum_operator(H::HamiltonOperator) = momentum_operator(H.elementgrid)
momentum_operator(H::HamiltonOperatorFreeParticle) = momentum_operator(H.elementgrid)


## Projection operator

struct ProjectionOperator <: AbstractOperator{1}
    state::QuantumState
end

function (po::ProjectionOperator)(qs::QuantumState)
    s = bracket(po.state,qs)
    return s * po.state
end

get_elementgrid(po::ProjectionOperator) = get_elementgrid(po.state)
Unitful.unit(po::ProjectionOperator) = unit(po.state)


## Density operator

"""
    density_operator(qs::QuantumState)
    density_operator(sd::SlaterDeterminant, occupations::Int=2)
    density_operator(sd::SlaterDeterminant, occupations::AbstractVector)

Return probability density.

For Slater determinants occupation numbers can be changed, for all or individual orbitals.
"""
density_operator(qs::QuantumState) = ScalarOperator(get_elementgrid(qs), ketbra(qs, qs))
function density_operator(sd::SlaterDeterminant, occupations::Int=2)
    if occupations == 1 
        return sum( density_operator, sd.orbitals )
    else
        return occupations * sum( density_operator, sd.orbitals )
    end
end


function density_operator(sd::SlaterDeterminant, occupations::AbstractVector)
    @assert length(sd.orbitals) == length(occupations)
    sum( x -> x[2]*density_operator(x[1]), zip(sd.orbitals, occupations) )
end

function charge_density(qs; charge=-1u"e_au")
    @argcheck dimension(charge) == dimension(u"C")
    return density_operator(qs) * charge
end


function electric_potential(cdensity::ScalarOperator, ct::AbstractCoulombTransformation; correction=true, showprogress=false)
    @argcheck dimension(cdensity) == dimension(u"C")
    if unit(cdensity) != u"e_au"
        tmp = auconvert(cdensity)
        if correction
            ϕ = poisson_equation(tmp.vals, ct, tmax=ct.tmax, showprogress=showprogress)
        else
            ϕ = poisson_equation(tmp.vals, ct, showprogress=showprogress)
        end
    else
        if correction
            ϕ = poisson_equation(cdensity.vals, ct, tmax=ct.tmax, showprogress=showprogress)
        else
            ϕ = poisson_equation(cdensity.vals, ct, showprogress=showprogress)
        end
    end
    return ScalarOperator(get_elementgrid(cdensity), ϕ; unit=u"hartree/e_au") 
end


function electric_potential(qs, ct::AbstractCoulombTransformation; charge=-1u"e_au", correction=true, showprogress=false)
    ρ = charge_density(qs; charge=charge)
    return electric_potential(ρ, ct; showprogress=showprogress, correction=correction)
end

function electric_potential(qs; charge=-1u"e_au", showprogress=false)
    ct = optimal_coulomb_tranformation(get_elementgrid(qs))
    return electric_potential(qs, ct; charge=charge, showprogress=showprogress)
end