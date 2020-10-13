using TensorOperations
using ProgressMeter
using OffsetArrays
using SpecialFunctions




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


function transformation_tensor_alt(elements::CubicElements, gpoints, w, t; δ=1)
    @assert length(w) == length(gpoints)
    @assert δ > 0
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
                    rmax = (r + δ*maximum(abs, (βp - αm, βm - αp) ))*t[p]
                    rmin = (r - δ*minimum(abs, (βp - αm, βm - αp) ))*t[p]
                    T[α,β,I,J,p] = w[β] * 0.5*√π*erf(rmin, rmax)/(rmax - rmin)
                end
            end
        end
    end
    return T
end


function transformation_harrison_alt(elements::CubicElements, gpoints, w, nt::Int;
                                tmax=10, μ=1, δ=1)
    @assert length(w) == length(gpoints)
    @assert tmax > 0
    @assert δ > 0
    T = similar(gpoints,
        length(gpoints),
        length(gpoints),
        elements.npoints,
        elements.npoints,
        nt
    )
    ele = getcenters(elements)
    s, ws = gausspoints(nt; elementsize=(-tmax,tmax))
    es = elementsize(elements)/2
    off = OffsetArray( vcat(-es, gpoints, es), 0:length(gpoints)+1)
    Threads.@threads for p ∈ eachindex(s)
        for (I,J) ∈ Iterators.product(eachindex(ele), eachindex(ele))
            for β ∈ 1:length(gpoints)
                for α ∈ 1:length(gpoints)
                    r = off[β] - off[α] + ele[J] - ele[I]
                    βp = 0.5*(off[β+1] - off[β])
                    βm = 0.5*(off[β-1] - off[β])
                    αp = 0.5*(off[α+1] - off[α])
                    αm = 0.5*(off[α-1] - off[α])
                    rmax = (r + δ*maximum(abs, (βp - αm, βm - αp) ))
                    rmin = (r - δ*minimum(abs, (βp - αm, βm - αp) ))

                    # Mean value of T-tensor in r=0±δr
                    meanval = erf(rmin*exp(s[p]), rmax*exp(s[p]))/(rmax - rmin)
                    T[α,β,I,J,p] = w[β] * 0.5*√π*exp(-0.25μ^2*exp(-2s[p]))*meanval
                end
            end
        end
    end
    return T, s, ws
end


function transformation_harrison(elements::CubicElements, gpoints, w, nt::Int;
                                tmax=10, μ=1)
    @assert length(w) == length(gpoints)
    @assert tmax > 0
    T = similar(gpoints,
        length(gpoints),
        length(gpoints),
        elements.npoints,
        elements.npoints,
        nt
    )
    ele = getcenters(elements)
    s, ws = gausspoints(nt; elementsize=(-tmax,tmax))
    Threads.@threads for p ∈ eachindex(s)
        for (I,J) ∈ Iterators.product(eachindex(ele), eachindex(ele))
            for β ∈ 1:length(gpoints)
                for α ∈ 1:length(gpoints)
                    r = gpoints[β] - gpoints[α] + ele[J] - ele[I]
                    e = exp( -r^2*exp(2s[p]) - 0.25μ^2*exp(-2s[p]) + s[p] )
                    T[α,β,I,J,p] = w[β]*e
                end
            end
        end
    end
    return T, s, ws
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

function density_harrison(elements, gpoints, μ)
    ρ = density_tensor(elements, gpoints)
    r = similar(ρ)
    ne = length(elements)
    np = length(gpoints)
    for (I, J, K) ∈ Iterators.product(1:ne, 1:ne, 1:ne)
        for (i, j, k) ∈ Iterators.product(1:np, 1:np, 1:np)
            r[i,j,k,I,J,K] = sqrt((elements[I]+gpoints[i])^2+(elements[J]+gpoints[j])^2+(elements[K]+gpoints[k])^2)
        end
    end
    return ρ.*exp.(μ.*r), ρ.*exp.(-μ.*r)
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
