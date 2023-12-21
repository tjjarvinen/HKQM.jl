module QuantumSystem
    import LinearAlgebra: dot, ⋅
    import LinearAlgebra: cross, ×
    using Unitful
    using UnitfulAtomic
    using ..HelmholtzKernel

    export AbstractOperator

    export DerivativeOperator
    export HamiltonOperator
    export HamiltonOperatorFreeParticle
    export HamiltonOperatorMagneticField
    export LaplaceOperator
    export QuantumState
    export ScalarOperator
    export VectorOperator
    
    export braket
    export gradient_operator
    export ketbra
    export momentum_operator
    export normalize!
    export position_operator
    export vector_potential


    include("quantum_states.jl")
    include("operators.jl")
end