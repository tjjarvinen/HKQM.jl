using TensorOperations
using ProgressMeter


function transformation_tensor(elements, gpoints, w, t)
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
                        -t[p]^2*(gpoints[α]+elements[I]-gpoints[β]-elements[J])^2
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

function density_tensor(elements, gpoints, r)
    @assert length(r) == 3
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
                -(gpoints .+ elements[I] .- r[1]).^2 .-(gpoints[i]+elements[J]-r[2])^2 .-(gpoints[j]+elements[K]-r[3])^2
            )
        end
    end
    return ρ
end



function coulomb_tensor(ρ, transtensor, gpoints, wgp, t, wt)
    @assert length(gpoints) == length(wgp)
    @assert length(t) == length(wt)

    # Coulomb tensor (returned at end)
    ν = similar(ρ)
    ν .= 0

    # Initializing temporary tensors
    T = similar(transtensor, size(transtensor)[1:end-1])
    d = similar(ρ)
    e = similar(ρ)
    v = similar(ρ)

    @showprogress "Calculating v-tensor..." for p in Iterators.reverse(eachindex(wt))
        T = transtensor[:,:,:,:,p]
        @tensoropt begin
            d[α',β',γ, I',J',K] = T[γ,γ',K,K']*ρ[α',β',γ',I',J',K'];
            e[α',β,γ,I',J,K] = d[α',β',γ,I',J',K]*T[β,β',J,J'];
            v[α,β,γ,I,J,K] = e[α',β,γ,I',J,K]*T[α,α',I,I'];
        end
        ν = ν .+ wt[p].*v
    end
    return ν
end
