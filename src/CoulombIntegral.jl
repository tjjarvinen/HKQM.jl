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

"""
    self_energy(n_elements, n_gaussp, n_tpoints; tmax=10) -> Float64

Calculates self energy using cubic elemens and Gaussian quadrature.

# Arguments
- `n_elements` : number of elements
- `n_gaussp`   : number of Gauss points
- `n_tpoints`  : number of Gauss points in exponential function integration

# Keywords
- `tmax=10`  :  integration limit for exponential function
"""
function self_energy(n_elements, n_gaussp, n_tpoints; tmax=10)
    @info "Initializing elements and Gauss points"
    eq = CubicElements(-10, 10, n_elements)

    x, w = gausspoints(eq, n_gaussp)
    t, wt = gausspoints(n_tpoints; elementsize=(0,tmax))
    centers = getcenters(eq)

    @info "Generating transformation tensor"
    T = transformation_tensor(centers, x, w, t)

    @info "Generating electron density tensor"
    ρ = density_tensor(centers, x)

    # To run on GPU the previous ternsors need to be moved to GPU
    # eg.
    # cx = CuArray(x) and so on

    v = coulomb_tensor(ρ, T, x, w, t, wt; yi=1:n_elements, zi=1:n_elements)
    V = v .+ (π/100).*ρ


    ω = hcat([w for i in 1:n_elements]...)

    cc = V.*ρ
    E = @tensor ω[α,I]*ω[β,J]*ω[γ,K]*cc[α,β,γ,I,J,K]

end


end
