module CoulombIntegral

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
       momentum_operator,
       position_operator,
       ω_tensor

# Types
export AbstractQuantumState,
       DerivativeOperator,
       GradientOperator,
       HamiltonOperator,
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

# Benzene carbons
const C6=[
(1.548314974868006, -2.24845283486348, 0.10613968032401824),
 (-2.682787487347467, -0.15842197712358957, 0.5086519117871446),
 (2.6764758020912174, 0.1533828334396625, -0.5269180045077774),
 (1.1287957752010853, 2.409722062339043, -0.6629782854808328),
 (-1.55839477401676, 2.243426163371976, -0.14330416812658342),
 (-1.1124194086050785, -2.399630924833542, 0.7184115116206051),
]

end
