module CoulombIntegral

using GaussQuadrature
using OffsetArrays
using Polynomials
using ProgressMeter
using SpecialFunctions
using StaticArrays
using TensorOperations
using Unitful
using UnitfulAtomic

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
       EM_Hamilton,
       elementsize,
       gaussian_coulomb_integral,
       gaussiandensity_self_energy,
       gausspoints,
       getcenter,
       getcenters,
       grid1d,
       Hamilton,
       integrate,
       kinetic_energy,
       loglocalct,
       normalize!,
       poisson_equation,
       poisson_equation!,
       potential_energy,
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

# Functions
export bracket,
       helmholz_equation,
       helmholz_equation!,
       magnetic_current,
       momentum_operator,
       position_operator,
       ⋆,
       vector_potential,
       ω_tensor

# Types
export AbstractQuantumState,
       DerivativeOperator,
       GradientOperator,
       HamiltonOperator,
       HamiltonOperatorFreeParticle,
       HamiltonOperatorMagneticField,
       LaplaceOperator,
       OperatorSum,
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
