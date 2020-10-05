using TensorOperations
using ProgressMeter
using OffsetArrays
using SpecialFunctions
using FastGaussQuadrature




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

function transformation_tensor_gauss_laguerre(elements::CubicElements, ngpoints, ntpoints)
    t, wt = gausslaguerre(ntpoints)
    x, w = gausspoints(elements, ngpoints)
    T = zeros(
        length(x),
        length(x),
        elements.npoints,
        elements.npoints,
        length(t)
    )
    ele = getcenters(elements)
    for p ∈ eachindex(t)
        for (I,J) ∈ Iterators.product(eachindex(ele), eachindex(ele))
            for β ∈ eachindex(x)
                for α ∈ eachindex(x)
                    T[α,β,I,J,p] = w[β].*exp.(t[p]
                        -t[p]^2*(x[α]+ele[I]-x[β]-ele[J])^2
                    )
                end
            end
        end
    end
    return T, x,w, t,wt
end


function transformation_tensor_alt(elements::CubicElements, gpoints, w, t)
    @assert length(w) == length(gpoints)
    T = similar(gpoints,
        length(gpoints),
        length(gpoints),
        elements.npoints,
        elements.npoints,
        length(t)
    )
    s = elementsize(elements)/2
    ele = getcenters(elements)
    off = OffsetArray( vcat(-s, gpoints, s), 0:length(gpoints)+1)
    Threads.@threads for p ∈ eachindex(t)
        for (I,J) ∈ Iterators.product(eachindex(ele), eachindex(ele))
            for β ∈ 1:length(gpoints)
                for α ∈ 1:length(gpoints)
                    r = off[β] - off[α] + ele[J] - ele[I]
                    βp = 0.5*(off[β+1] - off[β])
                    βm = 0.5*(off[β-1] - off[β])
                    αp = 0.5*(off[α+1] - off[α])
                    αm = 0.5*(off[α-1] - off[α])
                    rmax = (r + maximum(abs, (βp - αm, βm - αp) ))*t[p]
                    rmin = (r - minimum(abs, (βp - αm, βm - αp) ))*t[p]
                    T[α,β,I,J,p] = (w[β]/(rmax - rmin)) * 0.5*√π*erf(rmin, rmax)
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
