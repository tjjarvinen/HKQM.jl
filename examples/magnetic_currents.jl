using HKQM

# Load visualizations
include(joinpath(pkgdir(HKQM), "examples", "visualize_wave_function.jl"))



ceg = ElementGridSymmetricBox(2.5u"Å", 4, 32)

# Positions of Hydrogen atoms
r₁ = [0.37, 0., 0.] .* 1u"Å"
r₂ = [-0.37, 0., 0.] .* 1u"Å"

# Nunclear potential
V₁ = nuclear_potential_harrison_approximation(ceg, r₁, "H")
V₂ = nuclear_potential_harrison_approximation(ceg, r₂, "H")
V = V₁ + V₂

H = HamiltonOperator(V)

# Initial state
ϕ = particle_in_box(ceg, 1,1,1)
ψ = SlaterDeterminant( ϕ )

# Solve wave_function
ψ1 = scf(ψ, H)

# Define magnetic field
B = [0., 0., 10.0].*u"T"  # 10T in z-direction
A = vector_potential(ceg, B...)

Hm = HamiltonOperatorMagneticField(V,A)

# Solve system in magnetic field
# starting form non-magnetic field calculation
ψm = scf(ψ1, Hm)


# total magnetic current
j = magnetic_current(ψm, Hm)

# para magnetic current
jp = para_magnetic_current(ψm)

# dia magnetic current
jd = j - jp


## Plot magnetic current

# 2D plot of xy-plane at z=0
plot_current(j; z=0)

# 3D plot
# ng is number of lines per dimension
plot_current(j; mode_3d=true, ng=10)

# 2D plot of paramagnetic current (z=0)
plot_current(jp)