using HKQM
using Interpolations
using GLMakie
# or
#using WGLMakie 
#WGLMakie.activate!(; fps=30)

##

"""
    get3d(psi::QuantumState)
    get3d(sop::ScalarOperator)

Transform 6D tensor form to 3D euclidean coordinates.
Returns `Array{T,3}`.
"""
function get3d(psi::QuantumState)
    tmp = permutedims(psi.psi, (1,4,2,5,3,6))
    s = size(tmp)
    return reshape(tmp, (s[1]*s[2], s[3]*s[4], s[5]*s[6]))
end


function get3d(sop::ScalarOperator)
    tmp = permutedims(sop.vals, (1,4,2,5,3,6))
    s = size(tmp)
    return reshape(tmp, (s[1]*s[2], s[3]*s[4], s[5]*s[6]))
end


function get_coordinates(ceg; convert_to=u"Å")
    a = ustrip(convert_to, 1u"bohr")
    tx = get_1d_grid(ceg, 1) .* a
    ty = get_1d_grid(ceg, 2) .* a
    tz = get_1d_grid(ceg, 3) .* a
    x = reshape(tx, (length(tx)) )
    y = reshape(ty, (length(ty)) )
    z = reshape(tz, (length(tz)) )
    return x, y, z
end


function get_interpolator(psi::QuantumState)
    x, y, z = get_coordinates(HKQM.get_elementgrid(psi))
    Psi3d = get3d(psi)
    return LinearInterpolation((x,y,z), Psi3d)
end


##

function Base.minimum(ceg::AbstractElementGrid)
    return ceg[ firstindex(ceg) ] 
end


function Base.maximum(ceg::AbstractElementGrid)
    return ceg[ lastindex(ceg) ]
end

##

"""
    plot_wave_function(psi::QuantumState; Kwargs)

Plots wave function using Makie. `mask` is used to remove very small values
from plot. It is percentage relative to maximum and minimum values that are
removed from plot. You might need to adjust it to get a good figure.

# Keywords
- `levels=10`   :  both positive and negative values are discretized to this many different colours
- `mask=0.3`    :  mask values that have relative absolute value smaller than positive or negative extremum 
- `alpha=0.05`  :  transparency of plot - smaller means more transparent
- `n_points=30` :  number of interpolation points used per dimentsion in the figure
- `resolution=(800, 800)`  :  resolution of the returned figure
"""
function plot_wave_function(psi::QuantumState; levels=10, mask=0.3, alpha=0.05, n_points=30, resolution=(800, 800))
    tmin = ustrip.( u"Å", minimum(HKQM.get_elementgrid(psi)) .* u"bohr" )
    tmax = ustrip.( u"Å", maximum(HKQM.get_elementgrid(psi)) .* u"bohr" )

    x = LinRange(tmin[1], tmax[1], n_points)
    y = LinRange(tmin[2], tmax[2], n_points)
    z = LinRange(tmin[3], tmax[3], n_points)

    w = get_interpolator(psi)
    vol = [w(ix,iy,iz) for ix in x, iy in y, iz in z]

    vmax = maximum(vol)
    vmin = minimum(vol)

    v1 = LinRange(vmin, mask*vmin, levels)
    v2 = LinRange(mask*vmax, vmax, levels)

    if vmin < 0 && vmax > 0
        v = vcat(v1, v2)
    elseif vmin < 0 # vmax < 0
        # Only negative values are plotted
        v = vcat(v1, -1.0 .* v1 ) 
    else # 0 < vmin < vmax
        # Only positive values are plotted
        v = vcat( -1.0 .* v2, v2 )
    end

    cmap = :Hiroshige
    fig = Figure(resolution = resolution)
    ax = Axis3(fig[1,1]; perspectiveness = 0.5, azimuth = 6.62,
        elevation = 0.57)
    contour!(ax, x, y, z, vol; colormap = cmap, alpha = alpha, levels = v)
    return fig
end