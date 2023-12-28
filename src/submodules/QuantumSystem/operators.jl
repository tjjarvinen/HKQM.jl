abstract type AbstractOperator{N} end
abstract type GridFixedOperator{N} <: AbstractOperator{N} end
abstract type AbstractCompositeOperator{N} <: AbstractOperator{N} end
abstract type AbstractHamiltonOperator <: AbstractOperator{1} end

get_elementgrid(::AbstractOperator) = missing
get_elementgrid(ao::GridFixedOperator) = ao.elementgrid
Base.size(::AbstractOperator) = missing
Base.size(ao::GridFixedOperator) = size(get_elementgrid(ao))
Base.length(::AbstractOperator{N}) where N = N
Base.firstindex(::AbstractOperator) = 1
Base.lastindex(ao::AbstractOperator) = length(ao)

function Base.show(io::IO, ao::AbstractOperator{N}) where N
    print(io, "Operator with $N dimensions")
end


function Base.show(io::IO, ao::GridFixedOperator{N}) where N
    s = size(ao)
    print(io, "Operator with $N dimensions and grid size $s")
end

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

Unitful.unit(::AbstractOperator) = missing
Unitful.unit(ao::GridFixedOperator) = ao.unit
Unitful.dimension(ao::GridFixedOperator) = dimension(unit(ao))

function Unitful.uconvert(u::Unitful.Units, op::GridFixedOperator)
    @assert dimension(u) == dimension(op)
    unit(op) == u && return op
    T = (eltype ∘ eltype)(get_elementgrid(op)) # make sure type is correct
    conv = (T ∘ ustrip ∘ uconvert)(u, 1*unit(op)) * u / unit(op)
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

function cross(v1::AbstractOperator{3}, v2::AbstractVector)
    @assert length(v2) == 3
    r1 = v1[2]*v2[3] - v1[3]*v2[2]
    r2 = v1[1]*v2[3] - v1[3]*v2[1]
    r3 = v1[1]*v2[2] - v1[2]*v2[1]
    return [r1,r2,r3]
end

Base.:(*)(op::AbstractOperator{1}, qs::QuantumState) = op(qs)
Base.:(*)(op::AbstractOperator, qs::QuantumState) = map(f->f(qs), op)


braket(psi1::QuantumState, ao::AbstractOperator{1}, psi2::QuantumState) = braket(psi1, ao*psi2)

function braket(psi1::QuantumState, ao::AbstractOperator, psi2::QuantumState)
    return map( o->braket(psi1, o*psi2), ao )
end

## ScalarOperator

struct ScalarOperator{TA, T, D} <: GridFixedOperator{1} where{TA<:AbstractArray{T, D}, T, D<:Int}
    elementgrid::AbstractElementGrid
    vals::TA
    unit::Unitful.FreeUnits
    function ScalarOperator(ega::AbstractElementGrid{<:Any, D}, vals::AbstractArray{T, D}; unit=NoUnits) where {T,D}
        new{typeof(vals), eltype(vals), D}(ega, vals, unit)
    end
end

function ScalarOperator(so::ScalarOperator; unit=NoUnits)
    return ScalarOperator(get_elementgrid(so), so.vals; unit=unit)
end

function ScalarOperator(qs::QuantumState; unit=NoUnits)
    return ScalarOperator(get_elementgrid(qs), qs.psi; unit=unit)
end

get_values(so::ScalarOperator) = so.vals

function convert_array_type(T, so::ScalarOperator)
    return ScalarOperator(get_elementgrid(so), T(so.vals); unit=unit(so) )
end

function (so::ScalarOperator{<:Any,<:Any,D})(qs::QuantumState{<:Any,<:Any,D}) where D
    @assert size(so) == size(qs)
    return QuantumState(get_elementgrid(so), so.vals.*qs.psi, unit(so)*unit(qs))
end


function Base.map(f, so::ScalarOperator; output_unit=unit(so))
    return ScalarOperator(
        get_elementgrid(so),
        f.(so.vals);
        unit=output_unit
    )    
end


function Base.map(
    f, 
    so1::ScalarOperator{<:Any, <:Any, D}, 
    so2::ScalarOperator{<:Any, <:Any, D}; 
    output_unit=unit(so1)
) where D
    @assert size(so1) == size(so2)
    return ScalarOperator(
        get_elementgrid(so1),
        f.(so1.vals, so2.vals);
        unit=output_unit
    )    
end


function Base.:(+)(
    so1::ScalarOperator{<:Any, T1, D}, 
    so2::ScalarOperator{<:Any, T2, D}
) where {T1, T2, D}
    @assert size(so1) == size(so2)
    @assert dimension(so1) == dimension(so2)
    T = promote_type(T1, T2)
    conv = ustrip(unit(so1), one(T)*unit(so2))
    return map(so1, so2) do a,b
        muladd(conv, b, a)
    end
end

function Base.:(-)(
    so1::ScalarOperator{<:Any, T1, D}, 
    so2::ScalarOperator{<:Any, T2, D}
) where {T1, T2, D}
    @assert size(so1) == size(so2)
    @assert dimension(so1) == dimension(so2)
    T = promote_type(T1, T2)
    conv = -ustrip(unit(so1), one(T)*unit(so2))
    return map(so1, so2) do a,b
        muladd(conv, b, a)
    end
end

for OP in (:(+), :(-))
    @eval begin
        function Base.$OP(a::ScalarOperator, x::Number)
            @assert dimension(a) == dimension(x) "Dimension missmatch"
            y = ustrip(uconvert(unit(a), x))
            return map( b->$OP(b,y), a)
        end
        function Base.$OP(x::Number, a::ScalarOperator)
            @assert dimension(a) == dimension(x) "Dimension missmatch"
            y = ustrip(uconvert(unit(a), x))
            return map( b-> $OP(y,b), a)
        end
    end
end


for OP in (:(/), :(*))
    @eval begin
        function Base.$OP(a::ScalarOperator,b::ScalarOperator)
            return map( (x,y)-> $OP(x,y), a, b; output_unit=$OP(unit(a), unit(b)))
        end
        function Base.$OP(a::ScalarOperator, x::Number)
            return map( b-> $OP(b,ustrip(x)), a; output_unit=$OP(unit(a), unit(x)) )
        end
        function Base.$OP(x::Number, a::ScalarOperator)
            return map( b-> $OP(ustrip(x),b), a; output_unit=$OP(unit(x), unit(a)) )
        end
    end
end


# Functions that need unitless input
for op in (:sin, :cos, :tan, :exp, :log)
    @eval function Base.$op(so::ScalarOperator)
        @assert dimension(so) == dimension(NoUnits) "Operator needs to be dimensionless"
        return map($op, so)
     end
end

# Functions that can change units
for op in (:sqrt, :cbrt)
    @eval Base.$op(so::ScalarOperator) = map($op, so; output_unit=$op(unit(so)))
end

# Functions that dont change units
for op in (:real, :imag)
    @eval Base.$op(so::ScalarOperator) = map($op, so; output_unit=unit(so))
end

Base.:(^)(a::ScalarOperator, x::Number) = map( y->y^x, a; output_unit=unit(a)^x)

function Unitful.uconvert(a::Unitful.Units, so::ScalarOperator{<:Any, T, <:Any}) where T
    @assert dimension(a) == dimension(so) "Dimension missmatch"
    a == unit(so) && return so
    conv = (T ∘ ustrip ∘ uconvert)(a, 1*unit(so))
    return map( x->conv*x, so; output_unit=a)
end


## VectorOperator

struct VectorOperator{N} <: AbstractOperator{N}
    operators::Vector{AbstractOperator{1}}
    function VectorOperator(op::AbstractVector{<:AbstractOperator{1}})
        @assert all(x-> size(x)===size(op[begin]), op) "Scalar operators have different sizes"
        @assert all(x-> dimension(x)==dimension(op[begin]), op)
        new{length(op)}(op)
    end
end

VectorOperator(op::AbstractOperator{1}...) = VectorOperator( collect(op) )
Base.getindex(v::VectorOperator, i...) = v.operators[i...]

Unitful.unit(v::VectorOperator) = unit(v[begin])
Unitful.dimension(v::VectorOperator) = dimension(v[begin])


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
            VectorOperator( map( x-> $OP(x,s), v) )
        end
        function Base.$OP(s::ScalarOperator,v::VectorOperator)
            @assert size(v) == size(s)
            VectorOperator( map( x-> $OP(s,x), v) )
        end
    end
end


for OP in [:(/)]
    @eval begin
        function Base.$OP(v::VectorOperator,s::ScalarOperator)
            @assert size(v) == size(s)
            VectorOperator( map( x-> $OP(x,s), v) )
        end
        Base.$OP(a::VectorOperator, x::Number) = VectorOperator($OP.(a.operators,x))
    end
end


Base.:(-)(a::VectorOperator) = VectorOperator( map(x->-x, a)  )


## Position operator

function position_operator(ega::ElementGridArray, i)
    tmp = map( j-> ega.r[i][j[i]],  eachindex(ega))
    return ScalarOperator(ega, tmp; unit=unit(ega))
end


function position_operator(ega::ElementGridArray)
    tmp = [ position_operator(ega, i) for i in 1:length(ega.r) ]
    return VectorOperator(tmp)
end


## Derivative operators


struct DerivativeOperator <: AbstractOperator{1}
    coordinate::Int
    order::Int
end

function Base.show(io::IO, ao::DerivativeOperator)
    print(io, "DerivativeOperator for $(ao.coordinate) dimension and order $(ao.order)")
end


function (devo::DerivativeOperator)(psi::QuantumState)
    ega = get_elementgrid(psi)
    if devo.coordinate == 1
        tmp = derivative_x(ega, psi.psi; order=devo.order)
    elseif devo.coordinate == 2
        tmp = derivative_y(ega, psi.psi; order=devo.order)
    elseif devo.coordinate == 3
        tmp = derivative_z(ega, psi.psi; order=devo.order)
    else
        error("not implented yet")
    end
    return QuantumState(ega, tmp, unit(psi)/unit(ega)^devo.order)
end


Unitful.dimension(devo::DerivativeOperator) = dimension(u"m"^-devo.order)




function gradient_operator(ega::AbstractElementGrid{<:Any, D}) where D
    g = [ DerivativeOperator(i, 1) for i in 1:D ]
    return VectorOperator(g)    
end


function momentum_operator(ega::AbstractElementGrid)
    return -im * u"ħ_au" * gradient_operator(ega)
end



struct LaplaceOperator <: AbstractOperator{1} end

function Base.show(io::IO, ::LaplaceOperator)
    print(io, "Laplace Operator")
end

function (lo::LaplaceOperator)(psi::QuantumState)
    ega = get_elementgrid(psi)
    tmp = laplacian(get_elementgrid(psi), psi.psi)
    return QuantumState(ega, tmp, unit(psi)/unit(ega)^2)
end


Unitful.dimension(lo::LaplaceOperator) = dimension(u"m"^-2)


## Untility operators

struct ConstantTimesOperator{TO,T,N} <: AbstractOperator{N}
    op::TO
    a::T
    function ConstantTimesOperator(op::AbstractOperator{N}, a=1) where N
        if ustrip(imag(a)) ≈ 0
            new{typeof(op),typeof(real(a)), N}(op, real(a))
        else
            new{typeof(op),typeof(a), N}(op, a)
        end
    end
end


Base.:(*)(op::AbstractOperator, a::Number) = ConstantTimesOperator(op,a)
Base.:(*)(a::Number, op::AbstractOperator) = ConstantTimesOperator(op,a)
Base.:(/)(op::AbstractOperator, a::Number) = ConstantTimesOperator(op,1/a)

Base.getindex(cto::ConstantTimesOperator,i::Int) = cto.a*cto.op[i]

Unitful.unit(op::ConstantTimesOperator) = unit(op.op) * unit(op.a)
Unitful.dimension(op::ConstantTimesOperator) = dimension(op.op) * dimension(op.a)

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






struct OperatorSum{TO1, TO2, N} <: AbstractCompositeOperator{N}
    op1::TO1
    op2::TO2
    function OperatorSum(op1::AbstractOperator{N}, op2::AbstractOperator{N}) where N
        @assert dimension(op1) === dimension(op2) "Operators have different unit dimensions"
        new{typeof(op1),typeof(op2),N}(op1, op2)
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
Unitful.dimension(sos::OperatorSum) = dimension(sos.op1)

Base.getindex(op::OperatorSum, i::Int) = op.op1[i] + op.op2[i]





struct OperatorProduct{TO1, TO2} <: AbstractCompositeOperator{1}
    op1::TO1
    op2::TO2
    function OperatorProduct(op1::AbstractOperator{1}, op2::AbstractOperator{1})
        new{typeof(op1),typeof(op2)}(op1, op2)
    end
end

function Base.:(*)(op1::AbstractOperator{1}, op2::AbstractOperator{1})
    return OperatorProduct(op1,op2)
end

(op::OperatorProduct)(qs::QuantumState) = op.op1(op.op2(qs))

Unitful.unit(op::OperatorProduct) = unit(op.op1) * unit(op.op2)
Unitful.dimension(op::OperatorProduct) = dimension(op.op1) * dimension(op.op2)



## Hamiltonians


Unitful.dimension(AbstractHamiltonOperator) = dimension(u"J")


struct HamiltonOperatorFreeParticle{T, TU} <: AbstractHamiltonOperator
    ∇²::LaplaceOperator
    m::T
    output_unit::TU
    function HamiltonOperatorFreeParticle(; m=1u"me_au", output_unit=u"hartree")
        @assert dimension(m) == dimension(u"kg")
        @assert dimension(output_unit) == dimension(u"J")
        new{typeof(m), typeof(output_unit)}(LaplaceOperator(), m, output_unit)
    end
end

Unitful.unit(H::HamiltonOperatorFreeParticle) = H.output_unit

function (H::HamiltonOperatorFreeParticle)(qs::QuantumState)
    a = u"ħ_au"^2/(-2H.m)
    return a * (H.∇²*qs) |> unit(H)
end



struct HamiltonOperator{T,TU,TV} <: AbstractHamiltonOperator
    T::HamiltonOperatorFreeParticle{T, TU}
    V::TV
    function HamiltonOperator(V::AbstractOperator{1}; m=1u"me_au", output_unit=u"hartree")
        @assert dimension(V) == dimension(u"J")
        @assert dimension(m) == dimension(u"kg")
        T = HamiltonOperatorFreeParticle(; m=m, output_unit=output_unit)
        new{typeof(T.m),typeof(output_unit),typeof(V)}(T, V)
    end
end

Unitful.unit(H::HamiltonOperator) = unit(H.T)


(H::HamiltonOperator)(qs::QuantumState) = H.T*qs + H.V*qs




struct HamiltonOperatorMagneticField{T,TU,TV,TA,Tq} <: AbstractHamiltonOperator
    T::HamiltonOperatorFreeParticle{T, TU}
    V::TV
    A::TA
    q::Tq
    function HamiltonOperatorMagneticField(
            V::AbstractOperator{1},
            A::AbstractOperator{3};
            m=1u"me_au",
            q=-1u"e_au",
            output_unit=u"hartree"
        )
        @assert dimension(V) == dimension(u"J")
        @assert dimension(A) == dimension(u"T*m")
        @assert dimension(m) == dimension(u"kg")
        @assert dimension(q) == dimension(u"e_au")
        T = HamiltonOperatorFreeParticle(; m=m, output_unit=output_unit)
        new{typeof(T.m),typeof(output_unit),typeof(V),typeof(A),typeof(q)}(T, V, A, q)
    end
end


Unitful.unit(H::HamiltonOperatorMagneticField) = unit(H.T)


function (H::HamiltonOperatorMagneticField)(ψ::QuantumState)
    #TODO make this better
    p = momentum_operator(H.A[1])
    ϕ = H.q^2*(H.A⋅H.A)*ψ
    ϕ += H.q*(p⋅H.A)*ψ
    ϕ += H.q*(H.A⋅p)*ψ
    ϕ /= 2H.T.m
    ϕ += H.V*ψ
    return H.T*ψ + ϕ
end


momentum_operator(so::ScalarOperator) = momentum_operator(get_elementgrid(so))
momentum_operator(H::HamiltonOperator) = momentum_operator(H.V)

function momentum_operator(H::HamiltonOperatorMagneticField)
    p = momentum_operator(H.A[1])
    return p + H.q*H.A
end


"""
    vector_potential(ceg, Bx, By, Bz)

Gives vector potential for constant magnetic field.
"""
function vector_potential(eg, Bx, By, Bz)
    @assert dimension(Bx) == dimension(By) == dimension(Bz) == dimension(u"T")
    r = position_operator(eg)
    # A = r×B/2
    Ax = (r[2]*Bz-r[3]*By)/2
    Ay = (r[3]*Bx-r[1]*Bz)/2
    Az = (r[1]*By-r[2]*Bx)/2
    return VectorOperator(Ax, Ay, Az)
end