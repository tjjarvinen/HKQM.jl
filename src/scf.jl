
function normalize!(ψ, ceg::CubicElementGrid)
    N = integrate(ψ, ceg, ψ)
    ψ .*= 1/sqrt(N)
    return ψ
end


struct Hamilton{NE, NG}
    ceg::CubicElementGrid
    dt::Matrix{Float64}
    V::Array{Float64, 6}
    mass::Float64
    function Hamilton(ceg::CubicElementGrid, dt, V; mass=1)
        @assert size(dt)[1] == size(ceg)[1]
        @assert size(ceg) == size(V)
        s = size(ceg)
        new{s[4],s[1]}(ceg, dt, V, mass)
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

function potential_energy(h::Hamilton, ψ)
    return integrate(ψ, h.ceg, h.V.*ψ)
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
