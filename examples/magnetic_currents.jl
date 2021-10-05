using HKQM
using Rotations
using Interpolations
using WGLMakie # Or some other Makie backend


## Helper functions to initialize the system


# Generates initial state that can be optimized in decent time
function initial_state(V::ScalarOperator)
    q = QuantumState(V.elementgrid, exp.(-V.vals))
    normalize!(q)
    return q
end

function benzene_like_potential(ceg; α=0.5u"bohr^-2", depth=1u"hartree")
    # Generate benzene carbon locations
    xyz = [139., 0, 0] .* u"pm"
    rot = RotZ(π/3) # Rotation matrix
    benz = [xyz]
    for i in 1:5
        push!(benz, rot*benz[end])
    end

    # Generate potential
    r = position_operator(ceg)
    V = sum(benz) do x
        q = r + x
         return -depth * exp(-α*(q⋅q))
    end
    return V
end

## Plotting tools

# Change from elment grid (6D) to regural grid (3D)
function get3d(ψ)
    s = size(ψ)
    a = s[1]*s[end]
    q = Array{Float64}(undef,a,a,a)
    for I in 1:s[4], J in 1:s[5], K in 1:s[6]
        q[(1:s[1]).+(I-1)*s[1], (1:s[2]).+(J-1)*s[2], (1:s[3]).+(K-1)*s[3]] =  ψ[:,:,:,I,J,K]
    end
    return q
end


function plot_potential(op::ScalarOperator; title="Potential Energy [Hartree]", levels=15)
    X = vec(xgrid(op.elementgrid)) |> collect
    X =  X.* u"bohr"
    x = uconvert.(u"Å", X)
    z = length(x)/2 |> Int
    ρ = get3d(op.vals)
    f = Figure()
    Axis(f[1,1], xlabel="x [Å]", ylabel="y [Å]", title=title )
    co = contourf!(ustrip(x), ustrip(x), ρ[:,:,z]; levels=levels, colormap=:plasma)
    Colorbar(f[1, 2], co)
    return f
end

function plot_density(ψ::QuantumState; title="Electron Density", levels=15)
    function get3d_density(ψ::QuantumState)
        ρ = conj!(ψ).*ψ
        return get3d(real.(ρ))
    end
    X = vec(xgrid(ψ.elementgrid)) |> collect
    X =  X.* u"bohr"
    x = uconvert.(u"Å", X)
    z = length(x)/2 |> Int
    ρ = get3d_density(ψ)
    f = Figure()
    Axis(f[1,1], xlabel="x [Å]", ylabel="y [Å]", title=title )
    co = contourf!(ustrip(x), ustrip(x), ρ[:,:,z]; levels=levels, colormap=:plasma)
    Colorbar(f[1, 2], co)
    return f
end

function plot_current(ψ::QuantumState, j; z=0, n=32, mode_3d=false, ng=30, arrow_size=0.06, title="title")
    X = vec(xgrid(ψ.elementgrid)) |> collect
    x = ustrip.( uconvert.(u"Å", X.*u"bohr") )
    u = interpolate((x,x,x), get3d(j[1]), Gridded(Linear()))
    v = interpolate((x,x,x), get3d(j[2]), Gridded(Linear()))
    w = interpolate((x,x,x), get3d(j[3]), Gridded(Linear()))

    f(rx,ry,rz) = Point3(
        u(rx,ry,rz),
        v(rx,ry,rz),
        w(rx,ry,rz)
    )
    f(rx,ry) = Point2(
        u(rx,ry,z),
        v(rx,ry,z)
    )
    xmin = minimum(x)
    xmax = maximum(x)
    r = range(xmin; stop=xmax, length=n)
    if mode_3d
        ngrid = min(ng, 15)
        fig = streamplot(f, r, r, r; colormap = :viridis, arrow_size = arrow_size, gridsize = (ngrid, ngrid))
        return fig
    else
        fig = Figure()
        Axis(fig[1,1], xlabel="x [Å]", ylabel="y [Å]", title=title )
        sp = streamplot!(f, r, r; colormap = :magma, arrow_size = arrow_size, gridsize = (ng, ng), title=title)
        return fig
    end
end

## No magnetic field calculation

# 2^3 elements with 64^3 Gauss points in element
ceg = CubicElementGrid(6u"Å", 2, 64)

# Well depth is 1 hartree. Well width is α, values 0.5-1.0 are ok.
V = benzene_like_potential(ceg; α=0.7u"bohr^-2", depth=3u"hartree")

H = HamiltonOperator(V)

ψ = initial_state(V)

# One orbital -> two electrons 
sd = SlaterDeterminant(ψ)

@info "Solving Restricted Hartree-Fock for two electrons."
@info "Doing a single Helmholtz equation. Progress bar shows how long it takes per iteration."
helmholtz_equation!(sd, H; showprogress=true)

@info "Doing 12 more Helmholtz iterations to get good convergence."
@info "Energy is printed, so that progress can be followed."
for i in 1:12
    helmholtz_equation!(sd, H)
end

 J = coulomb_operator(sd)
@info "Final total energy is $(bracket(H+0.5J,sd))"



## Magnetic field calculation

@info "Adding magnetic field to system."

B = [0., 0., 100.0].*u"T"
A = vector_potential(ceg, B...)
@info "Magnetic field is set to Bx = $(B[1]), By = $(B[2]) and Bz = $(B[3])"

Hm = HamiltonOperatorMagneticField(V,A)

@info "Doing a single Helmholtz equation in magnetic field. This will take longer due to complex numbers."
sdm = helmholtz_equation(sd, Hm; showprogress=true)

@info "Doing 5 more iterations. See energy for convergence"
for i in 1:5
    global sdm = helmholtz_equation(sdm, Hm)
end

Jm = coulomb_operator(sdm)
@info "Final energy in magnetic field is $(real(bracket(Hm+0.5Jm, sdm)))"

ψm = sdm.orbitals[1]


# Magnetic current
j = magnetic_current(ψm, Hm);

# Para magnetic current
j_para =  magnetic_current(ψm, H);

# Dia magnetic current
j_dia = map((x,y)-> x.-y, j,j_para);

## Plotting results


fpot = plot_potential(H.V)

fpsi = plot_density(ψ)


# 2d plots of magnetic currents
f = plot_current(ψm, j; title="Magnetic Current in Plane")
fp = plot_current(ψm, j_para; title="Para Magnetic Current in Plane")
fd = plot_current(ψm, j_dia; title="Dia Magnetic Current in Plane")

# 3d plot of paramagnetic current
fp = plot_current(ψm, j; mode_3d=true, ng=11)
