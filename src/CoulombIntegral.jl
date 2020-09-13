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
C6=[
(0.819333,  -1.18983,     0.0561667),
(-1.41967,   -0.0838333,   0.269167),
(1.41633,    0.0811667,  -0.278833),
(0.597333,  1.27517,    -0.350833),
(-0.824667,   1.18717,    -0.0758333),
(-0.588667, -1.26983,     0.380167),
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
"""
function self_energy(n_elements, n_gaussp, n_tpoints; tmax=10, atoms=C6)
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

    v = coulomb_tensor(ρ, T, x, w, t, wt)
    V = v .+ (π/tmax^2).*ρ   # add correction

    # Integraton weights fore elements + Gausspoints in tersor form
    ω = hcat([w for i in 1:n_elements]...)

    cc = V.*ρ
    E = @tensor ω[α,I]*ω[β,J]*ω[γ,K]*cc[α,β,γ,I,J,K]
    return E
end


end
