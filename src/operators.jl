
abstract type AbstractOperator{N} end

abstract type AbstractScalarOperator <: AbstractOperator{1} end
abstract type AbstractVectorOperator <: AbstractOperator{3} end
abstract type AbstractCompositeOperator{N} <: AbstractOperator{N} end
abstract type AbstractHamiltonOperator <: AbstractOperator{1} end

Base.size(ao::AbstractOperator) = size(ao.elementgrid)
Base.show(io::IO, ao::AbstractOperator) = print(io, "Operator size=$(size(ao))")
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

#Base.:(*)()

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

struct ScalarOperator{T} <: AbstractScalarOperator
    elementgrid::CubicElementGrid
    vals::T
    unit::Unitful.FreeUnits
    function ScalarOperator(ceg::CubicElementGrid, ψ::AbstractArray{<:Any,6}; unit::Unitful.FreeUnits=NoUnits)
        new{typeof(ψ)}(ceg, ψ, unit)
    end
end

(so::ScalarOperator)(ψ::AbstractArray{<:Any,6}) = QuantumState(so.elementgrid, so.vals.*ψ, unit(so))
function (so::ScalarOperator)(qs::QuantumState)
    @assert size(so) == size(qs)
    return QuantumState(so.elementgrid, so.vals.*qs.ψ, unit(so)*unit(qs))
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


for op in (:sin, :cos, :exp, :log, :sqrt, :cbrt, :real, :imag)
    @eval Base.$op(so::ScalarOperator) = ScalarOperator(so.elementgrid, $op.(so.vals); unit=$op(so.unit))
end

function Unitful.uconvert(a::Unitful.Units, so::ScalarOperator)
    @assert dimension(a) == dimension(so) "Dimension missmatch"
    a == unit(so) && return so
    conv = ustrip(uconvert(a, 1*unit(so)))
    return ScalarOperator(so.elementgrid, so.vals.*conv; unit=a)
end


## Vector Operator

struct VectorOperator{N} <: AbstractOperator{N}
    elementgrid::CubicElementGrid
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

function position_operator(ceg::CubicElementGrid)
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

struct DerivativeOperator{NG,N} <: AbstractOperator{1}
    elementgrid::CubicElementGrid
    dt::DerivativeTensor{NG}
    function DerivativeOperator(ceg::CubicElementGrid, coordinatate::Int=1)
        @assert coordinatate ∈ 1:3 "coordinate needs to be 1, 2 or 3"
        dt = DerivativeTensor(ceg)
        new{size(ceg)[1], coordinatate}(ceg, dt)
    end
    function DerivativeOperator(ceg::CubicElementGrid, dt::DerivativeTensor, coordinatate::Int=1)
        @assert coordinatate ∈ 1:3 "coordinate needs to be 1, 2 or 3"
        @assert size(ceg)[1] == size(dt)[1]
        new{size(ceg)[1], coordinatate}(ceg, dt)
    end
end


function (d::DerivativeOperator{<:Any,1})(qs::QuantumState)
    return QuantumState(d.elementgrid, operate_x(d.dt, qs.ψ), unit(qs)*unit(d) )
end

function (d::DerivativeOperator{<:Any,2})(qs::QuantumState)
    return QuantumState(d.elementgrid, operate_y(d.dt, qs.ψ), unit(qs)*unit(d) )
end

function (d::DerivativeOperator{<:Any,3})(qs::QuantumState)
    return QuantumState(d.elementgrid, operate_z(d.dt, qs.ψ), unit(qs)*unit(d) )
end


Unitful.unit(::DerivativeOperator) = u"bohr"^-1

## Gradient Operator
struct GradientOperator{NG} <: AbstractOperator{3}
    elementgrid::CubicElementGrid
    dt::DerivativeTensor{NG}
    function GradientOperator(ceg::CubicElementGrid)
        dt = DerivativeTensor(ceg)
        new{size(ceg)[1]}(ceg, dt)
    end
end


function (go::GradientOperator)(ψ::AbstractArray{<:Any,6})
    return [QuantumState(go.elementgrid, operate_x(go.dt,ψ)),
            QuantumState(go.elementgrid, operate_y(go.dt,ψ)),
            QuantumState(go.elementgrid, operate_z(go.dt,ψ))]
end

function (go::GradientOperator)(ψ::QuantumState)
    return go(ψ.ψ)
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

struct ConstantTimesOperator{TO,T,N} <: AbstractOperator{N}
    elementgrid::CubicElementGrid
    op::TO
    a::T
    function ConstantTimesOperator(op::AbstractOperator{N}, a=1) where N
        if ustrip(imag(a)) ≈ 0
            new{typeof(op),typeof(real(a)), N}(op.elementgrid, op, real(a))
        else
            new{typeof(op),typeof(a), N}(op.elementgrid, op, a)
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




## Scalar operator sum

struct OperatorSum{TO1, TO2, N} <: AbstractCompositeOperator{N}
    elementgrid::CubicElementGrid
    op1::TO1
    op2::TO2
    function OperatorSum(op1::AbstractOperator{N}, op2::AbstractOperator{N}) where N
        @assert size(op1) == size(op2)
        @assert dimension(op1) == dimension(op2) "Operators have different unit dimensions"
        if unit(op1) == unit(op2)
            new{typeof(op1),typeof(op2),N}(op1.elementgrid, op1, op2)
        else
            new{typeof(op1),typeof(op2),N}(op1.elementgrid, op1, uconvert(unit(op1), op2))
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

struct OperatorProduct{TO1, TO2} <: AbstractCompositeOperator{1}
    elementgrid::CubicElementGrid
    op1::TO1
    op2::TO2
    function OperatorProduct(op1::AbstractOperator{1}, op2::AbstractOperator{1})
        @assert size(op1) == size(op2)
        @assert length(op1) == length(op2)  "Operators have different length"
        new{typeof(op1),typeof(op2)}(op1.elementgrid, op1, op2)
    end
end

function Base.:(*)(op1::AbstractOperator{1}, op2::AbstractOperator{1})
    return OperatorProduct(op1,op2)
end

(op::OperatorProduct)(qs::QuantumState) = op.op1(op.op2(qs))

Unitful.unit(op::OperatorProduct) = unit(op.op1) * unit(op.op2)


## Scalar operator division

struct OperatorDivision{TO1, TO2} <: AbstractCompositeOperator{1}
    elementgrid::CubicElementGrid
    op1::TO1
    op2::TO2
    function OperatorDivision(op1::AbstractOperator{1}, op2::AbstractOperator{1})
        @assert size(op1) == size(op2)
        @assert length(op1) == length(op2)  "Operators have different length"
        new{typeof(op1),typeof(op2)}(op1.elementgrid, op1, op2)
    end
end

function Base.:(/)(op1::AbstractOperator{1}, op2::AbstractOperator{1})
    return OperatorDivision(op1,op2)
end

Unitful.unit(oo::OperatorDivision) = unit(op.op1) / unit(op.op2)


## Laplace Operator
struct LaplaceOperator <: AbstractOperator{1}
    elementgrid::CubicElementGrid
    g::GradientOperator
    function LaplaceOperator(g::GradientOperator)
        new(g.elementgrid, g)
    end
    function LaplaceOperator(ceg::CubicElementGrid)
        new(ceg, GradientOperator(ceg))
    end
end

Unitful.unit(lo::LaplaceOperator) = unit(lo.g)^2

Base.:(*)(g1::GradientOperator, g2::GradientOperator) = LaplaceOperator(g1)

dot(g1::GradientOperator, g2::GradientOperator) = LaplaceOperator(g1)

function (lo::LaplaceOperator)(qs::QuantumState)
    tmp = similar(qs.ψ, size(lo.g.dt))
    tmp .= lo.g.dt
    @tensor w[i,j]:=tmp[i,k]*tmp[k,j]
    @tensor ϕ[i,j,k,I,J,K]:= w[i,ii]*qs.ψ[ii,j,k,I,J,K] + w[j,jj]*qs.ψ[i,jj,k,I,J,K] + w[k,kk]*qs.ψ[i,j,kk,I,J,K]
    return QuantumState(qs.elementgrid, ϕ, unit(qs)*unit(lo.g)^2)
end


## Special operators


function momentum_operator(ceg::CubicElementGrid)
    c = - im * 1u"ħ_au"
    g = GradientOperator(ceg)
    return c*g
end

function kinetic_energy_operator(ceg::CubicElementGrid, m=1u"me_au")
    @assert dimension(m) == dimension(u"kg")
    c = -1u"hartree * bohr^2"/(2*austrip(m))
    return c * LaplaceOperator(ceg)
end

## Hamilton operator

struct HamiltonOperatorFreeParticle{T} <: AbstractHamiltonOperator
    elementgrid::CubicElementGrid
    ∇²::LaplaceOperator
    c::T
    function HamiltonOperatorFreeParticle(ceg::CubicElementGrid;m=1u"me_au")
        @assert dimension(m) == dimension(u"kg")
        c = -1u"hartree * bohr^2"/(2*austrip(m))
        new{typeof(c)}(ceg, LaplaceOperator(ceg), c)
    end
end

Unitful.unit(H::HamiltonOperatorFreeParticle) = unit(H.c)*unit(H.∇²)

(H::HamiltonOperatorFreeParticle)(qs::QuantumState) = H.c*(H.∇²*qs)


struct HamiltonOperator{TF,TV} <: AbstractHamiltonOperator
    elementgrid::CubicElementGrid
    T::HamiltonOperatorFreeParticle{TF}
    V::TV
    function HamiltonOperator(V::AbstractOperator{1}; m=1u"me_au")
        @assert dimension(V) == dimension(u"J")
        @assert dimension(m) == dimension(u"kg")
        T = HamiltonOperatorFreeParticle(V.elementgrid; m=m)
        new{typeof(T.c),typeof(V)}(V.elementgrid, T, V)
    end
end

Unitful.unit(H::HamiltonOperator) = unit( 1*unit(H.V) + 1*unit(H.T) )

(H::HamiltonOperator)(qs::QuantumState) = H.T*qs + H.V*qs


struct HamiltonOperatorMagneticField{TF,TV,TA,Ta} <: AbstractHamiltonOperator
    elementgrid::CubicElementGrid
    T::HamiltonOperatorFreeParticle{TF}
    V::TV
    A::TA
    q::Ta
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
        new{typeof(T.c),typeof(V),typeof(A),typeof(q)}(V.elementgrid, T, V, A, q)
    end
end

Unitful.unit(H::HamiltonOperatorMagneticField) = unit(H.T)

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
