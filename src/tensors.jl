using TensorOperations
using ProgressMeter


function transformation_tensor(elements, gpoints, w, t)
    @assert length(w) == length(gpoints)
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



function coulomb_tensor(ρ, transtensor, gpoints, wgp, t, wt;
                        xi=1:size(ρ)[4], yi=1:1, zi=1:1)
    @assert length(gpoints) == length(wgp)
    @assert length(t) == length(wt)

    # Initializing δ tensor
    lt = length(t)
    δ = similar(ρ,lt,lt,lt)
    δ .= 0
    for i ∈ 1:lt
        δ[i,i,i] = 1
    end

    # Coulomb tensor (returned at end)
    ν = similar(ρ)

    # Initializing temporary tensors
    Tx = similar(transtensor, size(transtensor)[vcat(1:4,end)])
    Ty = similar(Tx)
    Tz = similar(Tx)
    d = similar(ρ, length(t), size(ρ)[1:end-1]...)
    e = similar(ρ, size(d)[1:end-1]...)
    v = similar(ρ, size(e)[1:end-1]...)
    vv = similar(ρ, size(v)[2:end]...)

    @showprogress "Calculating..." for (I, J, K) ∈ Iterators.product(xi, yi, zi)
        Tx = transtensor[:,:,:,I,:]
        Ty = transtensor[:,:,:,J,:]
        Tz = transtensor[:,:,:,K,:]
        @tensoropt begin
            d[p,α',β',γ, I',J'] = Tz[p,γ,γ',K']*ρ[α',β',γ',I',J',K'];
            e[p,α',β,γ,I'] = d[p1,α',β',γ,I',J']*Ty[p2,β,β',J']*δ[p1,p2,p];
            v[p,α,β,γ] = e[p1,α',β,γ,I']*Tx[p2,α,α',I']*δ[p1,p2,p];
            vv[α,β,γ] = v[p,α,β,γ] * wt[p];
        end
        ν[:,:,:,I,J,K] =  vv
    end
    return ν
end
