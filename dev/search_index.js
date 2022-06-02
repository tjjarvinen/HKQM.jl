var documenterSearchIndex = {"docs":
[{"location":"guide/#Package-Guide","page":"Guide","title":"Package Guide","text":"","category":"section"},{"location":"guide/#Define-Grid-for-Calculations","page":"Guide","title":"Define Grid for Calculations","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"Start by defining grid used in calculations","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"using HKQM\n\nceg = ElementGridSymmetricBox(5u\"Å\", 4, 24)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"This creates a cubic box with with side lenght of 5 Å that are divided to 4 elements and 24 Gauss-Lagrange points for each element. Resulting in total of (4*24)^3=884736 points.","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"The grid is also an Array that can be used as one","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"typeof(ceg) <: AbstractArray","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"size(ceg)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"eltype(ceg)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"The values are x-, y- and z-coordinates of the grid point in bohr.","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"ceg[1,1,1,3,3,3]","category":"page"},{"location":"guide/#Operator-algebra","page":"Guide","title":"Operator algebra","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"Generate basic operators for the grid","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"r = position_operator(ceg)\np = momentum_operator(ceg)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Operators have units defined with Unitful.jl package","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"unit(r)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"unit(p)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"These operator are vector operator and have lenght defined","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"length(r)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Individual components can be accessed with indexing","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"x = r[1]\ny = r[2]\nz = r[3]\n\nnothing # hide","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Operators support basic algebra operations","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"r + r\n2 * r\nr + [1u\"bohr\", 2u\"Å\", 1u\"pm\"]\nr + 1u\"bohr\"\nr / 2\nx * y\n\nnothing # hide","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Units are checked for the operations and operations that do not make sense are prohibited","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"r + p","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Vector operations are supported","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"r² = r ⋅ r\nl  = r × p\n\nnothing # hide","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Common functions can be used also","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"sin(1u\"bohr^-1\" * x)\nexp(-1u\"bohr^-2\" * r²)\n\nnothing # hide","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Functions require that the input is unitless","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"exp(-r²)","category":"page"},{"location":"guide/#Quantum-States","page":"Guide","title":"Quantum States","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"Quantum states can be created from operators","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"ψ = QuantumState( exp(-2u\"bohr^-2\" * r²) )","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"States can be normalized","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"normalize!(ψ) ","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Complex conjugate can be taken with","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"ϕ = conj(ψ)\nconj!(ψ)\n\nnothing # hide","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Quantum states have linear algebra defined","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"2ψ - ψ","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Inner product can be calculated with bracket function","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"bracket(ψ, 2ψ) ","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Operators can be applied to quantum state by multiplication","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"x * ψ","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Vector operators return arrays of quantum state","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"r * ψ","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Quantum states have units","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"unit(ψ) ","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"unit( x * ψ ) ","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Other Unitful functions like dimension and uconvert are defined also.","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Expectational values of operators can be calculated with bracket funtion","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"bracket(ψ, x, ψ) ","category":"page"},{"location":"guide/#Slater-Determinant","page":"Guide","title":"Slater Determinant","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"Slater determinant is orthonormal set of quantum states","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"st = SlaterDeterminant([ψ, (1u\"bohr^-1\"*x)*ψ])","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Slater determinat is an array of orbitals represented by quantum states","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"length(st)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"st[1]","category":"page"},{"location":"guide/#Hamilton-Operator","page":"Guide","title":"Hamilton Operator","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"Hamilton operator is a special operator that is needed for Helmholtz Greens function. To create it you need to create potential energy operator first","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"V = -30u\"eV\" * exp(-0.1u\"bohr^-2\" * r²) \n\nH = HamiltonOperator(V)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Default mass is one electron mass and it can be customised with m keyword","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"H_2me = HamiltonOperator(V; m=2u\"me_au\")\n\nnothing # hide","category":"page"},{"location":"guide/#Solving-Eigen-States-of-a-Hamiltonian","page":"Guide","title":"Solving Eigen States of a Hamiltonian","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"You need to generate initial state for Hamiltonian that gives negative energy!","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"ψ = QuantumState( exp(-0.2u\"bohr^-2\" * r²) )\nnormalize!(ψ)\n\nbracket(ψ, H, ψ)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"After that Helmholtz Greens function can be used to generate better estimate for the lowest eigen state","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"ϕ, E = solve_eigen_states(H, ψ; max_iter=10, rtol=1E-6)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"You can add more states to the solution by giving more intial states","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"ψ111 = particle_in_box(ceg, 1,1,1)\nψ112 = particle_in_box(ceg, 1,1,2)\n\nϕ2, E2 = solve_eigen_states(H, ψ111, ψ112)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Once estimate is self consistent a true solution has been found.","category":"page"},{"location":"guide/#Solving-Hartree-Fock-equation","page":"Guide","title":"Solving Hartree-Fock equation","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"Hartree-Fock equation can be solver with scf command.","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"V = -100u\"eV\" * exp(-0.1u\"bohr^-2\" * r²) \nH = HamiltonOperator(V)\n\nψ₁ = QuantumState( exp(-0.2u\"bohr^-2\" * r²) )\nψ₂ = 1u\"bohr^-1\"*r[1]*QuantumState( exp(-0.2u\"bohr^-2\" * r²) )\nsd = SlaterDeterminant(ψ₁, ψ₂)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"To check that all eigen values are negative calculate Fock matrix and look for diagonal values.","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"fock_matrix(sd, H)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"After that you can solve Hartree-Fock equations","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"sd1 = scf(sd, H; tol=1E-6, max_iter=10)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"tol is maximum chance in orbital overlap untill convergence is archieved. max_iter is maximum iterations calculated.","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Hartree-Fock energy is calculated by calling hf_energy","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"hf_energy(sd1, H)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Orbital energies can be found from diagonal of Fock matrix","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"fock_matrix(sd1, H)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Check also that offdiagonal elements are insignificant to make sure the system real solution has been found.","category":"page"},{"location":"guide/#Approximate-Nuclear-Potential","page":"Guide","title":"Approximate Nuclear Potential","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"There is an approximation to nuclear potential defined in here J. Chem. Phys. 121, 11587 (2004). It allows approximate electronic structure calculations.","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Here is an example for Hydrogen molecule.","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Define nuclear positions","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"r₁ = [0.37, 0., 0.] .* 1u\"Å\"\nr₂ = [-0.37, 0., 0.] .* 1u\"Å\"","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"After that create nuclear potential","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"V₁ = nuclear_potential_harrison_approximation(ceg, r₁, \"H\")\nV₂ = nuclear_potential_harrison_approximation(ceg, r₂, \"H\")\n\nV = V₁ + V₂","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"and Hamiltonian","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"H = HamiltonOperator(V)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Create initial orbital","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"ϕ = particle_in_box(ceg, 1,1,1)\nψ = SlaterDeterminant( ϕ )","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Solve SCF equations","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"ψ1 = scf(ψ, H)","category":"page"},{"location":"accuracy/#Accuracy-Testing","page":"Accuracy Tests","title":"Accuracy Testing","text":"","category":"section"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"There are several ways on how to test accuracy of the calculations, which is nessary in order to know how good the results are.","category":"page"},{"location":"accuracy/#Accuracy-of-Poisson-equation-Greens-function","page":"Accuracy Tests","title":"Accuracy of Poisson equation Greens function","text":"","category":"section"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"To test accuracy of Poisson equation and Helmholtz equation use test_accuracy function","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"test_accuracy","category":"page"},{"location":"accuracy/#HKQM.test_accuracy","page":"Accuracy Tests","title":"HKQM.test_accuracy","text":"test_accuracy(a::Real, ne::Int, ng::Int, nt::Int; kwords) -> Dict\n\nTest accuracy on Gaussian charge distribution self energy.\n\nArguments\n\na::Real   : Simulation box size\nne::Int   : Number of elements per dimension\nng::Int   : Number of Gausspoints per dimension for r\nnt::Int   : Number of Gausspoints per dimension for t\n\nKeywords\n\ntmax=300          : Maximum t-value for integration\ntmin=0            : Minimum t-value for integration\nmode=:combination : Integration type - options :normal, :log, :loglocal, :local, :combination and :optimal\nδ=0.25            : Localization parameter for local integration types\ncorrection=true   : Correction to tmax-> ∞ integration - true calculate correction, false do not calculate\ntboundary=20      : Parameter for :combination mode. Switch to :loglocal for t>tboundary else use :log\nα1=1              : 1st Gaussian is exp(-α1*r^2)\nα2=1              : 2nd Gaussian is exp(-α2*r^2)\nd=0               : Distance between two Gaussian centers\n\n\n\n\n\n","category":"function"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"It can be called with","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"using HKQM","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"test_accuracy(5u\"Å\", 4, 24, 96)","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"Which gives accuracy for 5 Å box with 4 elements per dimension and 24^3 Gauss-Lagrange points per element.","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"The default mode uses optimal_coulomb_tranformation to calculate Poisson equation","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"optimal_coulomb_tranformation","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"This is usually good enough.","category":"page"},{"location":"accuracy/#Integration-tuning","page":"Accuracy Tests","title":"Integration tuning","text":"","category":"section"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"note: Work in progress\n","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"This is work in progress. More coming later...","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"To perform integration there is a so called t variable that has to be integrated from zero to infinity. The main contribution comes from small values.","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"Normal mode will use normal Gauss-Lagrange itegration\nLogarithmic will distribute the points in logarithmic fashion\nLocal mean that average value around the points is calculated. This can be used with normal and logarithmic spacing","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"Fanally there is correction for very large values.","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"To test the integral accuracy calculate without correction and choose how many points are used for t-integrarion nt.","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"In example to use logarithmic spacing to integrate from t=20 to t=70 one can use","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"test_accuracy(5u\"Å\", 4, 24, 24; mode=:log, tmin=20, tmax=70, correction=false)","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"From the output we can see that integral is heavily overestimated, see \"integration error\" from output.","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"To test the same aree with loglocal mode gives","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"test_accuracy(5u\"Å\", 4, 24, 24; mode=:loglocal, tmin=20, tmax=70, correction=false)","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"and we can see that the accuracy was considerably improved.","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"To build integral tensor from parts you can use AbstractHelmholtzTensor types","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"HelmholtzTensorLinear\nHelmholtzTensorLog\nHelmholtzTensorLocalLinear\nHelmholtzTensorLocalLog\nHelmholtzTensorCombination","category":"page"},{"location":"accuracy/","page":"Accuracy Tests","title":"Accuracy Tests","text":"To overload the default tensor for calculations you need to create (=redefine) optimal_coulomb_tranformation function that returns HelmholtzTensor.","category":"page"},{"location":"visualization/#Visualizing-Wave-Function","page":"Visualizing Wave Function","title":"Visualizing Wave Function","text":"","category":"section"},{"location":"visualization/","page":"Visualizing Wave Function","title":"Visualizing Wave Function","text":"There is an example on plotting wave function in examples folder on the package. You can eather look at it and make your of recipe or include it. By default it uses GLMakie. But you can also use WGLMakie, by loading it instead. Note that GLMakie is probably faster.","category":"page"},{"location":"visualization/","page":"Visualizing Wave Function","title":"Visualizing Wave Function","text":"To include it just type","category":"page"},{"location":"visualization/","page":"Visualizing Wave Function","title":"Visualizing Wave Function","text":"include( joinpath(pkgdir(HKQM), \"examples\", \"visualize_wave_function.jl\") )","category":"page"},{"location":"visualization/","page":"Visualizing Wave Function","title":"Visualizing Wave Function","text":"After that you can use e.g.","category":"page"},{"location":"visualization/","page":"Visualizing Wave Function","title":"Visualizing Wave Function","text":"ceg = ElementGridSymmetricBox(5u\"Å\", 4, 24)\npsi = particle_in_box(ceg, 2, 3, 1)\n\nplot_wave_function(psi; resolution=(800,800))","category":"page"},{"location":"#HKQM.jl","page":"Home","title":"HKQM.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"DocTestSetup = quote\n    using HKQM\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"Documentation for HKQM.jl","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Installed with Julia package manager. From the Julia REPL, type ] to enter the Pkg REPEL more and run","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add https://github.com/tjjarvinen/HKQM.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"To test install type","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> test HKQM","category":"page"},{"location":"#Running-with-CPU","page":"Home","title":"Running with CPU","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"With CPU there are two things that define how many threads are used Julia number of threads and BLAS number of threads. This option affects on how many cores tensor contractions are done.","category":"page"},{"location":"","page":"Home","title":"Home","text":"From these Julia number of threads can be defined on startup with -t option. It can be checked once started with command","category":"page"},{"location":"","page":"Home","title":"Home","text":"Base.Threads.nthreads()","category":"page"},{"location":"","page":"Home","title":"Home","text":"The second option is how many threads BLAS is using. This can be find out with command","category":"page"},{"location":"","page":"Home","title":"Home","text":"Base.LinearAlgebra.BLAS.get_num_threads()","category":"page"},{"location":"","page":"Home","title":"Home","text":"Setting up number of threads for BLAS is done with","category":"page"},{"location":"","page":"Home","title":"Home","text":"Base.LinearAlgebra.BLAS.set_num_threads(n)","category":"page"},{"location":"#Number-of-Processes","page":"Home","title":"Number of Processes","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Number of Processes defines how many instances of Poisson/Helmholtz Greens functions are run on parallel. That is how many orbitals are updated in parallel. You do not want this option to be higher than number of orbitals. Ideally number of orbitals can be divided by number or processes.","category":"page"},{"location":"","page":"Home","title":"Home","text":"When done on same computer Julia can be started with option -p that defines number of processes used in calculation. See documentation for details. Alternatively you can use Distributed package to start more processes","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Distributed\n\naddprocs(n)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Number of worker processes can found out by typing","category":"page"},{"location":"","page":"Home","title":"Home","text":"nworkers()","category":"page"}]
}
