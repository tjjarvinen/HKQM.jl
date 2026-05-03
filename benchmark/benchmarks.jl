using HKQM.HelmholtzKernel
using HKQM.QuantumSystem
using BenchmarkTools
using Unitful
using UnitfulAtomic
using LinearAlgebra



# Initial structures

ne = 4
np = 32
ev = ElementVector(5u"Å", ne)
egl = ElementGridVectorLegendre(ev, np)
eg = ElementGridArray(egl, egl, egl)


ψ = particle_in_box(eg, 1,2,3)
ψ2 = 2*ψ
r = position_operator(eg)
p = momentum_operator(eg)
Dx = DerivativeOperator(1,1)
Dy = DerivativeOperator(2,1)
Dz = DerivativeOperator(3,1)
go = gradient_operator(eg)
lo = LaplaceOperator()
H = HamiltonOperator( (r ⋅ r)*1u"hartree/Å^2" )
l = r × p
x = r[1]

## Benchmarks
SUITE = BenchmarkGroup()

SUITE["Operate"] = BenchmarkGroup([],
        "ScalarOperator" => BenchmarkGroup(),
        "VectorOperator" => BenchmarkGroup(),
        "Number" => BenchmarkGroup(),
        "QuantumState" => BenchmarkGroup(),
)

SUITE["QuantumState"] = BenchmarkGroup()
SUITE["Integration"] = BenchmarkGroup()


SUITE["Operate"]["ScalarOperator"]["multiply"] = @benchmarkable $(r[1]) * $(r[2])
SUITE["Operate"]["ScalarOperator"]["divide"] = @benchmarkable $(r[1]) / $(r[2])
SUITE["Operate"]["QuantumState"]["ScalarOperator"] = @benchmarkable x * ψ

SUITE["Operate"]["Number"]["divide"] = @benchmarkable x / 2
SUITE["Operate"]["Number"]["multiply"] = @benchmarkable 2* x 
SUITE["Operate"]["Number"]["power"] = @benchmarkable x^2

SUITE["Operate"]["VectorOperator"]["QuantumState"] = @benchmarkable r * ψ
SUITE["Operate"]["VectorOperator"]["dot product"] = @benchmarkable r ⋅ r
SUITE["Operate"]["VectorOperator"]["cross product"] = @benchmarkable r × r

SUITE["Operate"]["QuantumState"]["Derivative-x"] = @benchmarkable Dx * ψ
SUITE["Operate"]["QuantumState"]["Derivative-y"] = @benchmarkable Dy * ψ
SUITE["Operate"]["QuantumState"]["Derivative-z"] = @benchmarkable Dz * ψ

SUITE["Operate"]["QuantumState"]["GradientOperator"] = @benchmarkable go * ψ
SUITE["Operate"]["QuantumState"]["momentum"]= @benchmarkable p * ψ
SUITE["Operate"]["QuantumState"]["LaplaceOperator"] = @benchmarkable lo * ψ
SUITE["Operate"]["QuantumState"]["angularmomentum"] = @benchmarkable l * ψ
SUITE["Operate"]["QuantumState"]["Hamiltonian"] = @benchmarkable H * ψ

SUITE["QuantumState"]["ketbra"] = @benchmarkable ketbra($ψ, $ψ)
SUITE["QuantumState"]["addition"] = @benchmarkable ψ + ψ
SUITE["QuantumState"]["substraction"] = @benchmarkable ψ - ψ
SUITE["QuantumState"]["multiply"] = @benchmarkable 2 * ψ
SUITE["QuantumState"]["divide"] = @benchmarkable ψ/2

SUITE["Integration"]["Norm"] = @benchmarkable braket($ψ, $ψ)
SUITE["Integration"]["Overlap"] = @benchmarkable braket($ψ, $ψ2)
SUITE["Integration"]["ScalarOperator"] = @benchmarkable braket($ψ, $x, $ψ)
SUITE["Integration"]["VectorOperator"] = @benchmarkable braket($ψ, $r, $ψ)
SUITE["Integration"]["MomentumOperator"] = @benchmarkable braket($ψ, $p, $ψ)
SUITE["Integration"]["angularmomentum"] = @benchmarkable braket($ψ, $l, $ψ)
SUITE["Integration"]["Hamiltonian"] = @benchmarkable braket($ψ, $H, $ψ)
