
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


function helmholtz_equation(qs::QuantumState, k)
    ega = get_elementgrid(qs)
    k = ustrip(unit(ega)^-1, k)
    @argcheck k > 0
    tx = default_transformation_tensor(ega, 1)
    ty = default_transformation_tensor(ega, 2)
    tz = default_transformation_tensor(ega, 3)
    tx = HelmholtzTensor(tx, k)
    ty = HelmholtzTensor(ty, k)
    tz = HelmholtzTensor(tz, k)
    tmp = apply_transformation(qs.psi, tx, ty, tz)
    return QuantumState(ega, tmp, unit(qs)*unit(ega)^2)
end


function helmholtz_equation(qs::QuantumState, H::HamiltonOperator)
    normalize!(qs)
    E = (real ∘ braket)(qs,H,qs)
    T = eltype(H.V.vals) #NOTE might need to remove T for forward more AD
    k = sqrt(-T(2H.T.m) * E)/u"ħ_au" 
    ϕ = H.V * ( T(H.T.m*u"ħ_au^-2") * qs )
    ψ = helmholtz_equation(ϕ, k)
    normalize!(ψ)
    return ψ
end


function scf(qs::QuantumState, H::HamiltonOperator; max_iter=10, conv=1e-6)
    E₀ = braket(qs, H, qs)
    @info "Initial energy is $E₀"
    tmp = helmholtz_equation(qs, H)
    E = braket(tmp, H, tmp)
    rchange = round( (E-E₀)/E₀ |> abs; sigdigits=2)
    @info "i=1 Energy is $E  :  relative change is $rchange"
    if rchange > conv
        for i in 2:max_iter
            E₀ = E
            tmp = helmholtz_equation(tmp, H)
            E = braket(tmp, H, tmp)
            rchange = round( (E-E₀)/E₀ |> abs; sigdigits=2)
            @info "i=$i Energy is $E  :  relative change is $rchange"
            if rchange < conv
                break
            end
        end
    end
    return tmp
end


"""
    nuclear_potential_harrison_approximation(args; kwargs)

Calculates Harrison style nuclear potential.
See [J. Chem. Phys. 121, 11587 (2004)](https://doi.org/10.1063/1.1791051)
for reference.

# Args
- `ceg`                       : Element grid used in calculation
- `system`                    : AtomsBase system

# Kwargs
- `electron_charge=-1u"e_au"`   : Electron charge in further calculations
- `precision=1E-6`              : Desired precision
"""
function nuclear_potential_harrison_approximation(
    ega,
    system;
    electron_charge=-1u"e_au",
    precision=1E-6
)
    r_op = position_operator(ega)
    V = sum( 1:length(system) ) do i 
        Zᵢ = atomic_number(system, i)
        rᵢ = position(system, i)
        # Harrison's special term for scaling the potential
        # to desired accuracy
        c_param = cbrt( 0.00435 * precision / Zᵢ^5 )

        rr = r_op - rᵢ
        # These two dont have unit
        r² = ( rr ⋅ rr ) / (c_param*u"bohr")^2 |> auconvert  
        r  = sqrt(r²) + 1E-10  # Make sure that no zero division

        U = erf(r)/r + 1/(3*√π) * ( exp(-r²) + 16exp(-4r²) )
        1u"hartree" / c_param * (Z * austrip(electron_charge) ) * U
    end
    return V
end