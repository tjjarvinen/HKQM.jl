module HKQM

using Reexport

using FastGaussQuadrature
using LinearAlgebra: dot, ⋅
using LinearAlgebra: cross, ×
using Polynomials
using ProgressMeter
using SpecialFunctions
using StaticArrays
using TensorOperations
using Tullio
@reexport using PeriodicTable
@reexport using Unitful
@reexport using UnitfulAtomic

# Submodule
include("submodules/ReferenceStates.jl")
using .ReferenceStates

import LinearAlgebra.dot
import LinearAlgebra.cross
import LinearAlgebra.normalize!

export AbstractElementGrid,
       AbstractOperator,
       CubicElementGrid,
       CubicElement,
       CubicElements,
       coulomb_correction,
       coulomb_tensor,
       CoulombTransformation,
       CoulombTransformationCombination,
       CoulombTransformationLocal,
       CoulombTransformationLog,
       CoulombTransformationLogLocal,
       coulombtransformation,
       DerivativeTensor,
       Element1D,
       elementsize,
       gaussian_coulomb_integral,
       gaussiandensity_self_energy,
       gausspoints,
       getcenter,
       getcenters,
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
       cross, ×,
       dot, ⋅,
       helmholtz_equation,
       helmholtz_equation!,
       ketbra, ⋆,
       magnetic_current,
       momentum_operator,
       nuclear_potential,
       optimal_coulomb_tranformation,
       para_magnetic_current,
       poisson_equation,
       poisson_equation!,
       position_operator,
       test_nuclear_potential,
       vector_potential,
       ω_tensor

# Abstract Types
export AbstractQuantumState

# Concrete Types
export DerivativeOperator,
       GradientOperator,
       HamiltonOperator,
       HamiltonOperatorFreeParticle,
       HamiltonOperatorMagneticField,
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
       VectorOperator

include("elements.jl")
include("states.jl")
include("tensors.jl")
include("operators.jl")
include("integrations.jl")
include("accuracytests.jl")
include("scf.jl")



end
