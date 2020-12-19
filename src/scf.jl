

function normalize!(ψ, ceg::CubicElementGrid)
    N = integrate(ψ, ceg, ψ)
    ψ .*= 1/sqrt(N)
    return ψ
end

abstract type AbstractHamilton end

struct Hamilton{NE, NG, T} <: AbstractHamilton
    ceg::CubicElementGrid
    dt::Matrix{Float64}
    V::T
    mass::Float64
    function Hamilton(ceg::CubicElementGrid, dt, V::AbstractArray; mass=1)
        @assert size(dt)[1] == size(ceg)[1]
        @assert size(ceg) == size(V)
        s = size(ceg)
        new{s[4],s[1], typeof(V)}(ceg, dt, V, mass)
    end
end

Base.show(io::IO, h::Hamilton) = print(io,"Hamiltonian")

function integrate(h::Hamilton, ψ)
    N = integrate(ψ, h.ceg, ψ)
    E_kin = kinetic_energy(h,ψ)
    E_pot = integrate(ψ, h.ceg, h.V.*ψ)
    return (E_kin + E_pot)/N
end

function kinetic_energy(h::Hamilton, ψ)
    return kinetic_energy(h.ceg, h.dt, ψ)/h.mass
end

function potential_energy(h::AbstractHamilton, ψ)
    return integrate(ψ, h.ceg, h.V.*ψ)
end


struct EM_Hamilton{NE, NG, TV, TA} <: AbstractHamilton
    ceg::CubicElementGrid
    mt::MomentumTensor{NG}
    V::TV
    mass::Float64
    A::TA
    A2::TV
    function EM_Hamilton(ceg::CubicElementGrid, mt, V, A; mass=1)
        @assert size(mt)[1] == size(ceg)[1]
        @assert size(ceg) == size(V)
        s = size(ceg)
        A2 = @. A[:,:,:,:,:,:,1]^2 + A[:,:,:,:,:,:,2]^2 + A[:,:,:,:,:,:,3]^2
        new{s[4],s[1], typeof(V), typeof(A)}(ceg, mt, V, mass, A, A2)
    end
end

Base.show(io::IO, h::EM_Hamilton) = print(io,"Hamiltonian in magnetic field")

function kinetic_energy(h::EM_Hamilton, ψ)
    Eₖ = kinetic_energy(h.ceg, Array(h.mt.dt), ψ)
    Eₐ = integrate(ψ, h.ceg, h.A2.*ψ)/137.03599908330935^2
    return (Eₖ + Eₐ)/h.mass
end


function _kinetic_middle_term(h::EM_Hamilton, ψ)
    ψint = similar(ψ)
    ψint .= 0
    tmp = similar(ψ)
    tmp = h.A[:,:,:,:,:,:,1].*ψ
    dt = Array(h.mt.dt)
end


function jacobi_iterations(ceg::CubicElementGrid, h::Hamilton, ψ0; n=10)
    ψ = deepcopy(ψ0)
    normalize!(ψ, ceg);
    E = [integrate(h, ψ)]
    @info "i=0  E=$(E[end])"
    for i in 1:n
        ct = loglocalct(ceg, 96; k=sqrt(-2E[end]));
        ψ = coulomb_tensor(h.V.*ψ, ct; tmax=300);
        normalize!(ψ, ceg);
        push!(E, integrate(h,ψ))
        @info "i=$i  E=$(E[end])"
    end
    return E[2:end], ψ
end

function helmholz_equation(ψ::QuantumState, H::HamiltonOperator;
                            tn=96, tmax=300, showprogress=false)
    normalize!(ψ)
    E = real(bracket(ψ,H,ψ))
    @info "E=$E"
    k  = sqrt( -2(austrip(E)) )
    ct = loglocalct(H.elementgrid, tn; k=k);
    ϕ = H.T.m*H.V*1u"ħ_au^-2" * ψ
    ϕ = poisson_equation(ϕ, ct; tmax=tmax, showprogress=showprogress);
    normalize!(ϕ)
    return ϕ
end

function helmholz_equation!(ψ::QuantumState, H::HamiltonOperator;
                            tn=96, tmax=300, showprogress=false)
    normalize!(ψ)
    E = real(bracket(ψ,H,ψ))
    @info "E=$E"
    k  = sqrt( -2(austrip(E)) )
    ct = loglocalct(H.elementgrid, tn; k=k);
    ϕ = H.T.m*H.V*1u"ħ_au^-2" * ψ
    ψ .= poisson_equation(ϕ, ct; tmax=tmax, showprogress=showprogress);
    normalize!(ψ)
    return ψ
end


function helmholz_equation(ψ::QuantumState, H::HamiltonOperatorMagneticField;
                            tn=96, tmax=300, showprogress=false)
    normalize!(ψ)
    E = real(bracket(ψ,H,ψ))
    @info "E=$E"
    k  = sqrt( -2(austrip(E)) )
    ct = loglocalct(H.elementgrid, tn; k=k);
    p = momentum_operator(H.T)
    ϕ = H.T.m*H.V*1u"ħ_au^-2" * ψ + (H.q^2*u"ħ_au^-2")*(H.A⋅H.A)*ψ
    ϕ += (H.q*u"ħ_au^-2")*(H.A⋅p + p⋅H.A) * ψ
    ϕ = poisson_equation(ϕ, ct; tmax=tmax, showprogress=showprogress);
    normalize!(ϕ)
    return ϕ
end
