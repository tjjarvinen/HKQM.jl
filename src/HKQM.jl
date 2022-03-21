module HKQM

using Reexport

using ArgCheck
using Distributed
using FastGaussQuadrature
using LinearAlgebra: dot, ⋅
using LinearAlgebra: cross, ×
using PolynomialBases
using Polynomials
using ProgressMeter
using SpecialFunctions
using StaticArrays
using TensorOperations
using Tullio
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
       elementsize,
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
       coulomb_operator,
       cross, ×,
       density_operator,
       dot, ⋅,
       exchange_operator,
       electric_potential,
       fock_matrix,
       get_1d_grid,
       get_center,
       helmholtz_equation,
       helmholtz_equation!,
       helmholtz_update,
       hf_energy,
       ketbra, ⋆,
       magnetic_current,
       momentum_operator,
       #nuclear_potential,
       optimal_coulomb_tranformation,
       overlap_matrix,
       para_magnetic_current,
       poisson_equation,
       poisson_equation!,
       position_operator,
       scf,
       scf!,
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
       ElementGridSymmetricBox,
       ElementGridVector,
       ElementGrid,
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
include("states.jl")
#include("bubbles.jl")
include("tensors.jl")
include("nuclearpotential.jl")
include("operators.jl")
include("integrations.jl")
include("accuracytests.jl")
include("scf.jl")
include("initial_states.jl")



end
