using Test
using HKQM
using Logging
using ForwardDiff
using SpecialFunctions


disable_logging(Logging.Info)




include("tests.jl")

include("test-tensor-ops.jl")