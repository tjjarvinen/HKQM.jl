module ToroidalCurrent

using Distributed
using Interpolations
using DelimitedFiles

using ..HKQM


export read_sysmoic, read_current
export toroidal_current, poloidal_current
export write_currents

function read_sysmoic(fname)
    J = Float64[]
    dJ = Float64[]

    x, y, z = open(fname, "r") do io

        # Find grid information
        line = readline(io)
        while( !occursin(r"for x=", line) )
            line = readline(io)
        end
        tmp = split(line)
        xmin = parse(Float64, tmp[3])
        xmax = parse(Float64, tmp[5])
        xstep = parse(Float64, tmp[7])
        line = readline(io)
        if !occursin(r"for y=", line)
            throw( ErrorException("File $fname does not have correct grid information") )
        end
        tmp = split(line)
        ymin = parse(Float64, tmp[3])
        ymax = parse(Float64, tmp[5])
        ystep = parse(Float64, tmp[7])
        line = readline(io)
        if !occursin(r"for z=", line)
            throw( ErrorException("File $fname does not have correct grid information") )
        end
        tmp = split(line)
        zmin = parse(Float64, tmp[3])
        zmax = parse(Float64, tmp[5])
        zstep = parse(Float64, tmp[7])
        
        # Skip to the beginning of data
        for _ in 1:15 
            readline(io)
            # add checks here
        end

        error_count = 0
        # Read data
        for (i,line) in zip( Iterators.cycle(1:12), eachline(io) )
            try
                tmp = parse.(Float64, split(line)[1:3] )
            catch 
                # SYSMOIC files have some error output
                # Set badly printed numbers to zero
                error_count += 1
                s = split(line)
                tmp = Float64[]
                for num in s
                    if occursin("E", num)
                        push!(tmp, parse(Float64, num) )
                    else
                        push!(tmp, 0.0)
                    end
                end
            end

            if i in 1:3
                append!(J, tmp)
            else
                append!(dJ, tmp)
            end
        end

        if error_count > 0
            @warn "Number of miss printed lines in SYSMOIC file is $error_count"
        end

        x = range(xmin, xmax; step=xstep )
        y = range(ymin, ymax; step=ystep )
        z = range(zmin, zmax; step=zstep )
        return x, y, z
    end

    # J is 3x3xN tensor
    # with N is based on grid information
    if length(J) != 3*3* length(x) * length(y) * length(z)
        throw( DimensionMismatch("SYSMOIC file read error, check file content for J") )
    end

    J = reshape(J, 3, 3, length(x), length(y), length(z) )
    J = permutedims(J, (2,1,3,4,5) )

    # Same for dJ
    if length(dJ) != 3*3*3 * length(x) * length(y) * length(z)
        ldJ = length(dJ)
        lref = 3*3*3 * length(x) * length(y) * length(z)
        throw( DimensionMismatch(
            "SYSMOIC file read error, check file content for dJ, length=$ldJ ref=$lref"
            ) )
    end
    
    dJ = reshape(dJ, 3, 3, 3, length(x), length(y), length(z) )
    dJ = permutedims(dJ, (2,1,3,4,5,6) )

    return Dict("J"=>J, "dJ"=>dJ, "x"=>x, "y"=>y, "z"=>z)
end

##
# For calculations we need to interpolate the data to HKQM type element grids

function give_J_interpolator(data, i, j )
    intp = LinearInterpolation( (data["x"], data["y"], data["z"]), data["J"][i,j,:,:,:])
    return intp
end

function give_dJ_interpolator(data, i, j, k)
    intp = LinearInterpolation( (data["x"], data["y"], data["z"]), data["dJ"][i,j,k,:,:,:])
    return intp
end


function get_current_component(ceg, data, i, j)
    intp = give_J_interpolator(data, i, j )
    vals = [ intp(r...) for r in ceg ]
    #NOTE unit is not correct
    return QuantumState(ceg, vals)
end

function get_derivative_component(ceg, data, i, j, k)
    intp = give_dJ_interpolator(data, i, j, k)
    vals = [ intp(r...) for r in ceg ]
    #NOTE unit is not correct
    return QuantumState(ceg, vals, u"bohr^-1")
end


##
# Read data to QuantumState type that can be operated on
function read_current(fname, ne=4, ng=32)
    @info "Reading file"
    data = read_sysmoic(fname)
    x = data["x"]
    Δx = maximum(x) - minimum(x)
    ceg = ElementGridSymmetricBox(Δx*1u"bohr", ne, ng)
    @info "Interpolating current"
    J = [ get_current_component(ceg, data, i,j)   for (i,j) in Iterators.product(1:3, 1:3) ]

    @info "Interpolating derivatives"
    dJ = [ get_derivative_component(ceg, data, i,j,k)   for (i,j,k) in Iterators.product(1:3, 1:3, 1:3) ]
    return J, dJ
end


function toroidal_current(dJ; B=[0.,0.,1.].*u"T")
    @assert size(dJ) == (3,3,3)
    # ∇²ψ = ∂Jₓ/∂y - ∂J_y/∂x
    # Is this correct for magnetic responce?
    ∇²ψ = [ dJ[1,i,2] - dJ[2,i,1] for i in 1:3 ]

    ∇ = GradientOperator( dJ[begin] )

    ψ = pmap( ∇²ψ ) do ρ
        poisson_equation(ρ) 
    end

    # J and dJ are unitless
    # So we need to adopt to it
    Bt = uconvert.(u"T", B)
    Bn = ustrip(Bt)

    # We have magnetic responce so ...
    J_toro = ∇×(Bn .* ψ)

    return J_toro
end


function poloidal_current(J; B=[0.,0.,1.].*u"T")
    @assert size(J) == (3,3)
    # ∇²ϕ = -J_z
    # Is this correct for magnetic responce?
    ∇²ϕ = [ -J[3,i] for i in 1:3 ]

    ∇ = GradientOperator( J[begin] )

    ϕ = pmap( ∇²ϕ ) do ρ
        poisson_equation(ρ) 
    end

    # J and dJ are unitless
    # So we need to adopt to it
    Bt = uconvert.(u"T", B)
    Bn = ustrip(Bt)

    # We have magnetic responce so ...
    J_polo = ∇×∇×(Bn .* ϕ)
    return J_polo
end


function write_currents(fname, J_toro, J_polo; n_points=20)
    # These are taken from ../examples/vizualizze_wave_function.jl
    function get3d(psi)
        tmp = permutedims(psi.psi, (1,4,2,5,3,6))
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
    function get_interpolator(psi)
        x, y, z = get_coordinates(HKQM.get_elementgrid(psi))
        Psi3d = get3d(psi)
        return LinearInterpolation((x,y,z), Psi3d)
    end

    ceg = get_elementgrid(J_toro[1])
    tmin = ustrip.( u"Å", minimum(ceg) .* u"bohr" )
    tmax = ustrip.( u"Å", maximum(ceg) .* u"bohr" )

    x = LinRange(tmin[1], tmax[1], n_points)
    y = LinRange(tmin[2], tmax[2], n_points)
    z = LinRange(tmin[3], tmax[3], n_points)

    j_toro = pmap(J_toro) do jᵢ
        w = get_interpolator(jᵢ)
        j = vec([ w(i,j,k) for i in x, j in y, k in z  ])
    end

    j_polo = pmap(J_polo) do jᵢ
        w = get_interpolator(jᵢ)
        j = vec([ w(i,j,k) for i in x, j in y, k in z  ])
    end

    wx = vec([ i for i in x, j in y, k in z  ])
    wy = vec([ j for i in x, j in y, k in z  ])
    wz = vec([ k for i in x, j in y, k in z  ])

    @info "Writing data to $fname"
    open(fname, "w") do io
        writedlm(io, [wx wy wz j_toro[1] j_toro[3] j_toro[3] j_polo[1] j_polo[2] j_polo[3]])
    end
    @info "Writing complite"
    @info "Distance is in Ångströms"
    @info "Collumn order is x, y, z, Jx_toro, Jy_toro, Jz_toro, Jx_polo, Jy_polo, Jz_polo"

    #return Dict("jt"=>j_toro, "jp"=>j_polo)
end

end