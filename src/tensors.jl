using TensorOperations
using ProgressMeter


function transformation_tensor(elements, gpoints, w, t; ϵ=1E-2)
    @assert length(w) == length(gpoints)
    T = similar(gpoints,
        length(gpoints),
        length(gpoints),
        length(elements),
        length(elements),
        length(t)
    )
    for p ∈ eachindex(t)
        for (I,J) ∈ Iterators.product(eachindex(elements), eachindex(elements))
            for β ∈ eachindex(gpoints)
                for α ∈ eachindex(gpoints)
                    T[α,β,I,J,p] = w[β].*exp.(
                        -t[p]^2*(gpoints[α]+elements[I]-gpoints[β]-elements[J]+ϵ)^2
                    )
                end
            end
        end
    end
    return T
end



function density_tensor(elements, gpoints)
    ρ = similar(elements,
        length(gpoints),
        length(gpoints),
        length(gpoints),
        length(elements),
        length(elements),
        length(elements)
    )
    ne = length(elements)
    np = length(gpoints)
    for (I, J, K) ∈ Iterators.product(1:ne, 1:ne, 1:ne)
        for (i, j) ∈ Iterators.product(1:np, 1:np)
            ρ[:,i,j,I,J,K] = exp.(
                -(gpoints .+ elements[I]).^2 .-(gpoints[i]+elements[J])^2 .-(gpoints[j]+elements[K])^2
            )
        end
    end
    return ρ
end

function density_tensor(elements, gpoints, r::AbstractVector)
    @assert length(r[1]) == 3
    ρ = similar(elements,
        length(gpoints),
        length(gpoints),
        length(gpoints),
        length(elements),
        length(elements),
        length(elements)
    )
    ρ .= 0
    ne = length(elements)
    np = length(gpoints)
    Threads.@threads for K ∈ 1:ne
        for (I, J) ∈ Iterators.product(1:ne, 1:ne)
            for (i, j) ∈ Iterators.product(1:np, 1:np)
                for xyz ∈ r
                    ρ[:,i,j,I,J,K] .+= exp.(
                        -(gpoints .+ elements[I] .- xyz[1]).^2 .-(gpoints[i]+elements[J]-xyz[2])^2 .-(gpoints[j]+elements[K]-xyz[3])^2
                    )
                end
            end
        end
    end
    return ρ
end



function coulomb_tensor(ρ, transtensor, gpoints, wgp, t, wt)
    @assert length(gpoints) == length(wgp)
    @assert length(t) == length(wt)

    # Coulomb tensor (returned at end)
    V = similar(ρ)
    V .= 0

    # Initializing temporary tensors
    #T = similar(transtensor, size(transtensor)[1:end-1])
    v = similar(ρ)

    @showprogress "Calculating v-tensor..." for p in Iterators.reverse(eachindex(wt))
        T = @view transtensor[:,:,:,:,p]
        @tensoropt v[α,β,γ,I,J,K] = T[α,α',I,I']*T[β,β',J,J']*T[γ,γ',K,K']*ρ[α',β',γ',I',J',K']

        V = V .+ wt[p].*v
    end
    return V
end
