using TensorOperations


function transformation_tensor(elements, gpoints, w, t, wt)
    @assert length(w) == length(gpoints)
    @assert length(t) ==length(wt)
    T = similar(gpoints,
        length(t),
        length(gpoints),
        length(gpoints),
        length(elements),
        length(elements)
    )

    for J ∈ eachindex(elements)
        for I ∈ eachindex(elements)
            for α ∈ eachindex(gpoints)
                for β ∈ eachindex(gpoints)
                    T[:,α,β,I,J] = w[β]*exp.(
                        -t.^2 .*(gpoints[α]+elements[I]-gpoints[β]-elements[J])^2
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
