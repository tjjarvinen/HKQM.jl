using HKQM
using AtomsBase

# Load visualizations
include(joinpath(pkgdir(HKQM), "examples", "visualize_wave_function.jl"))


eg = eg = HKQM.HelmholtzKernel.ElementGridVectorLegendre(2.5u"Å", 4, 32)
nceg = HKQM.HelmholtzKernel.ElementGridArray(eg,eg,eg)

# Positions of Hydrogen atoms
r₁ = [0.37, 0., 0.] .* 1u"Å"
r₂ = [-0.37, 0., 0.] .* 1u"Å"

sys = isolated_system( [Atom(:H,r₁), Atom(:H,r₂)])


# Nunclear potential
NV = HKQM.QuantumSystem.nuclear_potential_harrison_approximation(nceg, sys)

NH = HKQM.QuantumSystem.HamiltonOperator(NV)

# Initial state
nϕ = HKQM.QuantumSystem.particle_in_box(nceg, 1,1,1)
nψ = HKQM.QuantumSystem.SlaterDeterminant( nϕ )

# Solve wave_function
ψ1 = HKQM.QuantumSystem.scf(nψ, NH)

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
