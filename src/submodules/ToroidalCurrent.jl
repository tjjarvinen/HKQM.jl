module ToroidalCurrent

using Interpolations

using ..HKQM


export read_sysmoic, read_current

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

    if length(dJ) != 3*3*3 * length(x) * length(y) * length(z)
        ldJ = length(dJ)
        lref = 3*3*3 * length(x) * length(y) * length(z)
        throw( DimensionMismatch(
            "SYSMOIC file read error, check file content for dJ, length=$ldJ ref=$lref"
            ) )
    end
    
    dJ = reshape(dJ, 3, 3, 3, length(x), length(y), length(z) )

    return Dict("J"=>J, "dJ"=>dJ, "x"=>x, "y"=>y, "z"=>z)
end


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
    return QuantumState(ceg, vals)
end

function get_derivative_component(ceg, data, i, j, k)
    intp = give_dJ_interpolator(data, i, j, k)
    vals = [ intp(r...) for r in ceg ]
    return QuantumState(ceg, vals)
end


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

end