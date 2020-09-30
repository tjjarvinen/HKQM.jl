using TensorOperations


function gaussiandensity_self_energy(tmax, nt)
    _f(x) = (π/sqrt(2x^2+1))^3 
    t, w = gausspoints(nt; elementsize=(0.0, tmax))
    return sum(w .* _f.(t))
end 


function test_accuracy(n_elements, n_gaussp, n_tpoints; tmax=10, correction=true, alt=false, amax=10)
    @info "Initializing elements and Gauss points"
    eq = CubicElements(-amax, amax, n_elements)

    x, w = gausspoints(eq, n_gaussp)
    t, wt = gausspoints(n_tpoints; elementsize=(0,tmax))
    centers = getcenters(eq)

    @info "Generating transformation tensor"
    if alt
        T = transformation_tensor_alt(eq, x, w, t)
    else
        T = transformation_tensor(centers, x, w, t)
    end
    @info "Generating electron density tensor"

    ρ = density_tensor(centers, x)

    V = coulomb_tensor(ρ, T, x, w, t, wt)
    if correction
        V = V .+ (π/tmax^2).*ρ   # add correction
    end

    # Integraton weights fore elements + Gausspoints in tersor form
    ω = hcat([w for i in 1:n_elements]...)

    cc = V.*ρ
    E = @tensor ω[α,I]*ω[β,J]*ω[γ,K]*cc[α,β,γ,I,J,K]
    E_true = gaussiandensity_self_energy(500, 50000)
    @info "Calculated energy = $E"
    @info "True energy = $E_true"
    @info "Error = $(E-E_true)"
    return E-E_true
end