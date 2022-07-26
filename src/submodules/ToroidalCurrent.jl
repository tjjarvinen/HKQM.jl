module ToroidalCurrent
    
using ..HKQM


function read_sysmoic(fname)
    open(fname, "r") do io

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

        # Read data
        for (i,line) in zip( cycle(1:12), eachline(io) )
            tmp = parse.(Float64, split(line)[1:3] )
            if i == 1  # J_xx J_xy J_xz
                
            elseif i == 2
                
            elseif i == 3
            elseif i == 4
            elseif i == 5
            elseif i == 6
            elseif i == 7
            elseif i == 8
        end
    end
    
end

end