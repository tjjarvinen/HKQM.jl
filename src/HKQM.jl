module HKQM

using Reexport

using ArgCheck
using Distributed
using FastGaussQuadrature
using LinearAlgebra: dot, ⋅
using LinearAlgebra: cross, ×
using LinearAlgebra: mul!
using PolynomialBases
using Polynomials
using ProgressMeter
using SpecialFunctions
using StaticArrays
@reexport using PeriodicTable
@reexport using Unitful
@reexport using UnitfulAtomic
@reexport import LinearAlgebra: dot, cross, normalize!

# Submodule
include("submodules/ReferenceStates.jl")
using .ReferenceStates


export AbstractElementGrid,
       AbstractOperator,
       CubicElementGrid,
       CubicElement,
       coulomb_correction,
       DerivativeTensor,
       Element1D,
       gaussian_coulomb_integral,
       gaussiandensity_self_energy,
       gausspoints,
       grid1d,
       normalize!,
       self_energy,
       test_accuracy,
       test_accuracy_new,
       test_accuracy_ad,
       test_kinetic_energy,
       transformation_tensor,
       transformation_tensor_alt,
       transformation_harrison,
       transformation_tensor_alt,
       xgrid,
       ygrid,
       zgrid

# Methods
export bracket,
       charge_density,
       convert_array_type,
       convert_variable_type,
       coulomb_operator,
       cross, ×,
       density_operator,
       dot, ⋅,
       element_bounds,
       element_size,
       exchange_operator,
       electric_potential,
       fock_matrix,
       get_1d_grid,
       get_center,
       get_derivative_matrix,
       get_elementgrid,
       get_weight,
       helmholtz_equation,
       helmholtz_equation!,
       helmholtz_update,
       hf_energy,
       ketbra, ⋆,
       magnetic_current,
       momentum_operator,
       #nuclear_potential,
       nuclear_potential_harrison_approximation,
       optimal_coulomb_tranformation,
       overlap_matrix,
       para_magnetic_current,
       particle_in_box,
       poisson_equation,
       poisson_equation!,
       position_operator,
       scf,
       scf!,
       solve_eigen_states,
       test_nuclear_potential,
       vector_potential,
       ω_tensor

# Abstract Types
export  AbstractQuantumState,
        AbstractHelmholtzTensor,
        AbstractHelmholtzTensorSingle

# Concrete Types
export CubicElementArray,
       DerivativeOperator,
       ElementGridArray,
       ElementGridLegendre,
       ElementGridLobatto,
       ElementGridVector,
       ElementGridVectorLegendre,
       ElementGridVectorLobatto,
       ElementGridSymmetricBox,
       ElementVector,
       GradientOperator,
       HamiltonOperator,
       HamiltonOperatorFreeParticle,
       HamiltonOperatorMagneticField,
       HelmholtzTensorCombination,
       HelmholtzTensorLinear,
       HelmholtzTensorLocalLinear,
       HelmholtzTensorLocalLog,
       HelmholtzTensorLog,
       LaplaceOperator,
       NuclearPotentialTensor,
       NuclearPotentialTensorCombination,
       NuclearPotentialTensorGaussian,
       NuclearPotentialTensorLog,
       NuclearPotentialTensorLogLocal,
       OperatorSum,
       ProjectionOperator,
       QuantumState,
       ScalarOperator,
       SlaterDeterminant,
       VectorOperator


include("abstract_types.jl")
include("elements.jl")
include("deprecated.jl")
include("states.jl")
#include("bubbles.jl")
include("tensors.jl")
include("operators.jl")
include("nuclearpotential.jl")
include("integrations.jl")
include("accuracytests.jl")
include("scf.jl")
include("initial_states.jl")
include("submodules/ToroidalCurrent.jl")
include("submodules/HelmholtzKernel/HelmholtzKernel.jl")
include("submodules/QuantumSystem/QuantumSystem.jl")


end
