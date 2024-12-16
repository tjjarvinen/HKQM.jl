"""
    solve_eigen_states(H, initial_states::QuantumState...; max_iter=10, rtol=1E-6  )
    solve_eigen_states(H, sd::SlaterDeterminant; max_iter=10, rtol=1E-6 )

Solves Eigen states of an Hamiltonian Operator for given initial states.
Returns `Vector` holding eigen vectors and other holding eigen values.
Inital states can also be given as Slater determinant, where individual
orbitals are considered as initial states.
"""
function solve_eigen_states(H, initial_states::QuantumState...; max_iter=10, rtol=1E-6  )
    sd = SlaterDeterminant(initial_states...)
    return solve_eigen_states(H, sd; max_iter=max_iter, rtol=rtol )
end


function solve_eigen_states(H, sd::SlaterDeterminant; max_iter=10, rtol=1E-6 )
    function _energy(Ψ)
        E = pmap( Ψ ) do ϕ
            braket(ϕ, H, ϕ)
        end
        return E
    end
    
    E₀ = _energy(sd)
    E = []
    for i in 1:max_iter
        tmp = pmap( sd ) do ϕ
            helmholtz_equation(ϕ, H)
        end
        sd = SlaterDeterminant(tmp...)
        E = _energy(sd)
        ΔE = abs.( ( E .- E₀ ) ./ E₀ )
        if all( ΔE .< rtol )
            @info "Solution found in $i iterations"
            break
        else
            @info "i=$i  max relΔE= $(round(maximum(ΔE); sigdigits=2))"
            E₀ = E
        end
    end
    return collect(sd), E
end