# Visualizing Wave Function

There is an example on plotting wave function in `examples` folder on the package.
You can eather look at it and make your of recipe or include it.
By default it uses [GLMakie](https://makie.juliaplots.org/stable/documentation/backends/glmakie/).
But you can also use [WGLMakie](https://makie.juliaplots.org/stable/documentation/backends/wglmakie/),
by loading it instead. Note that GLMakie is probably faster.

To include it just type

```julia
include( joinpath(pkgdir(HKQM), "examples", "visualize_wave_function.jl") )
```

After that you can use e.g.

```julia
ceg = ElementGridSymmetricBox(5u"Å", 4, 24)
psi = particle_in_box(ceg, 2, 3, 1)

plot_wave_function(psi; resolution=(800,800))
```

## Magnetic current visualizations

Also visualizing magnetic current is supported

```
# J is magnetic current

# 3D plot
plot_current(J_toro; mode_3d=true)


# 2D plot on xy-plane with z=0
plot_current(J_toro; z=0)
```
