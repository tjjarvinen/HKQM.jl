
function particle_in_box(ceg, nx::Int, ny::Int, nz::Int)
    @assert nx > 0
    @assert ny > 0
    @assert nz > 0
    r = position_operator(ceg)
    Lx = element_size(ceg, 1)
    Ly = element_size(ceg, 2)
    Lz = element_size(ceg, 3)
    kx = nx * (π/Lx)
    ky = ny * (π/Ly)
    kz = nz * (π/Lz)
    tmp = sin( kx*(r[1] + Lx/2) ) * sin( ky*(r[2] + Ly/2) ) * sin( kz*(r[3] + Lz/2) )
    psi = QuantumState(ceg, tmp.vals)
    return normalize!(psi)
end