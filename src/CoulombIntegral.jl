module CoulombIntegral

export AbstractElementGrid,
       CubicElementGrid,
       CubicElement,
       CubicElements,
       coulomb_correction,
       coulomb_tensor,
       CoulombTransformation,
       coulombtransformation,
       Element1D,
       elementsize,
       gaussiandensity_self_energy,
       gausspoints,
       getcenter,
       getcenters,
       grid1d,
       self_energy,
       test_accuracy,
       test_accuracy_new,
       transformation_tensor,
       transformation_tensor_alt,
       transformation_harrison,
       transformation_tensor_alt,
       xgrid,
       ygrid,
       zgrid,
       ω_tensor


include("elements.jl")
include("tensors.jl")
include("accuracytests.jl")

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
