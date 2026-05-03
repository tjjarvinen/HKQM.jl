function accuracy_test(ega, alpha)
    r = position_operator(ega)
    r² = dot(r,r)
    ψ = QuantumState((2alpha/π)^(3//4) * exp( (-alpha/unit(r²)) * r²))
    ρ = charge_density(ψ)
    V = electric_potential(ρ)
    res = braket(ψ, V, ψ) |> austrip  # should be energy
    ref = - 2 * sqrt(alpha/π) * austrip(1.0u"bohr" / unit(ega))
    println("Calculate value is:",res)
    println("Reference is:", ref)
    println("Difference is:", res-ref)
    return res
end


function accuracy_test_galerkin(ega, alpha)
    r = position_operator(ega)
    r² = dot(r,r)

    α = alpha / unit(r²)

    ρ = (2α/π)^(3//2) * exp(-2α * r²)

    q = integrate(ega, ρ.vals)
    println("Integrated charge is: ", q)

    ρau = auconvert(ρ)

    tmp = similar(ρau.vals)
    apply_poisson!(tmp, ρau.vals, ega.Ttensor[1], ega.Ttensor[1], ega.Ttensor[1])

    # Galerkin energy: rho coefficients dotted with load vector
    res = sum(ρau.vals .* tmp)

    ref = 2 * sqrt(alpha / π)

    println("Calculated value is: ", res)
    println("Reference is: ", ref)
    println("Difference is: ", res - ref)

    return res
end

function accuracy_test_galerkin_raw(ega, alpha)
    r = position_operator(ega)
    r² = dot(r,r)

    α = alpha / unit(r²)

    ρ = (2α/π)^(3//2) * exp(-2α * r²)

    q = integrate(ega, ρ.vals)
    println("Integrated charge is: ", q)

    tmp = similar(ρ.vals)

    apply_poisson!(
        tmp,
        ρ.vals,
        ega.Ttensor[1],
        ega.Ttensor[1],
        ega.Ttensor[1],
    )

    res = sum(ρ.vals .* tmp)

    ref = 2 * sqrt(alpha / π)

    println("Calculated value is: ", res)
    println("Reference is: ", ref)
    println("Difference is: ", res - ref)

    return res
end