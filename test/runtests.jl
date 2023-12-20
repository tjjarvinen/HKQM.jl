using Test
#using HKQM
using Logging
using ForwardDiff
using SpecialFunctions


disable_logging(Logging.Info)




#include("tests.jl")

#@testset "TensorOperations" begin
#    include("test-tensor-ops.jl")
#end

#include("nuclear-potential.jl")

include("test-new.jl")
include("test-quantumsystem.jl")
