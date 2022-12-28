
function particle_in_box(ceg, nx::Int, ny::Int, nz::Int)
    @argcheck nx > 0
    @argcheck ny > 0
    @argcheck nz > 0
    r = position_operator(ceg)
    L = element_size(ceg)
    kx = nx * (π/L[1])
    ky = ny * (π/L[2])
    kz = nz * (π/L[3])
    tmp = sin( kx*(r[1] + L[1]/2) ) * sin( ky*(r[2] + L[2]/2) ) * sin( kz*(r[3] + L[3]/2) )
    psi = QuantumState(ceg, tmp.vals)
    return normalize!(psi)
end

function particle_in_box(T, ceg, nx::Int, ny::Int, nz::Int)
    tmp = particle_in_box(ceg, nx, ny, nz)
    return convert_array_type(T, tmp)
end


function atomic_orbital(ceg::AbstractElementGrid, Z::Real, r, n::Int, l::Int, ml::Int)
    #Z = elements[Symbol(aname)].number
    T = (eltype ∘ eltype)(ceg)
    ra = T.(austrip.(r)) .* u"bohr"
    TZ = T(Z)
    r0 = position_operator(ceg) - ra
    rr = sqrt(r0⋅r0)

    #NOTE this can give infinity with rr=0
    phi = exp(-(TZ/2)*u"bohr^-1"*rr)
    ϕ = QuantumState(ceg, phi.vals)
    return normalize!(ϕ)
end

function atomic_orbital(T, ceg::AbstractElementGrid, Z::Real, r, n::Int, l::Int, ml::Int)
    tmp = atomic_orbital(ceg, Z, r, n, l, ml)
    return convert_array_type(T, tmp)
end