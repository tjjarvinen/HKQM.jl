module QuantumSystem
    using ArgCheck
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
    export SlaterDeterminant
    export VectorOperator
    
    export braket
    export charge_density
    export density_operator
    export electric_potential
    export gradient_operator
    export helmholtz_equation
    export ketbra
    export momentum_operator
    export normalize!
    export particle_in_box
    export position_operator
    export scf
    export vector_potential


    include("quantum_states.jl")
    include("operators.jl")
    include("utils.jl")
    include("electronic_structure.jl")
end