module CoulombIntegral

export CubicElements,
       coulomb_tensor,
       elementsize,
       gausspoints,
       getcenters,
       self_energy,
       transformation_tensor


include("elements.jl")
include("tensors.jl")

# Benzene carbons
const C6=[
(1.548314974868006, -2.24845283486348, 0.10613968032401824),
 (-2.682787487347467, -0.15842197712358957, 0.5086519117871446),
 (2.6764758020912174, 0.1533828334396625, -0.5269180045077774),
 (1.1287957752010853, 2.409722062339043, -0.6629782854808328),
 (-1.55839477401676, 2.243426163371976, -0.14330416812658342),
 (-1.1124194086050785, -2.399630924833542, 0.7184115116206051),
]

"""
    self_energy(n_elements, n_gaussp, n_tpoints; tmax=10, atoms=C6) -> Float64

Calculates self energy using cubic elements and Gaussian quadrature.

# Arguments
- `n_elements` : number of elements
- `n_gaussp`   : number of Gauss points
- `n_tpoints`  : number of Gauss points in exponential function integration

# Keywords
- `tmax=10`    :  integration limit for exponential function
- `atoms=C6`   :  array of atom coordinates - default benzene carbons
- `correction=true`  : add correction to t-coordinate integration
"""
function self_energy(n_elements, n_gaussp, n_tpoints; tmax=10, atoms=C6, correction=true)
    @info "Initializing elements and Gauss points"
    eq = CubicElements(-10, 10, n_elements)

    @info "Atom positions:" atoms

    x, w = gausspoints(eq, n_gaussp)
    t, wt = gausspoints(n_tpoints; elementsize=(0,tmax))
    centers = getcenters(eq)

    @info "Generating transformation tensor"
    T = transformation_tensor(centers, x, w, t)
    @info "Generating electron density tensor"

    ρ = density_tensor(centers, x, atoms)

    # To run on GPU the previous ternsors need to be moved to GPU
    # eg.
    # cx = CuArray(x) and so on

    V = coulomb_tensor(ρ, T, x, w, t, wt)
    if correction
        V = V .+ (π/tmax^2).*ρ   # add correction
    end

    # Integraton weights fore elements + Gausspoints in tersor form
    ω = hcat([w for i in 1:n_elements]...)

    cc = V.*ρ
    E = @tensor ω[α,I]*ω[β,J]*ω[γ,K]*cc[α,β,γ,I,J,K]
    return E
end


end
