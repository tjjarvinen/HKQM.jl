# Visualizing Wave Function

There is an example on plotting wave function in `examples` folder on the package.
You can eather look at it and make your of recipe or include it.
By default it uses [WGLMakie](https://makie.juliaplots.org/stable/documentation/backends/wglmakie/).
But you can also use [GLMakie](https://makie.juliaplots.org/stable/documentation/backends/glmakie/),
by loading it instead.

To include it just type

```julia
include( joinpath(pkgdir(HKQM), "examples", "visualize_wave_function.jl") )
```

After that you can use e.g.

```julia
ceg = ElementGridSymmetricBox(5u"Å", 4, 24)
psi = particle_in_box(ceg, 2, 3, 1)

plot_psi(psi)
```
