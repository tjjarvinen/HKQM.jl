using Documenter
using HKQM

DocMeta.setdocmeta!(HKQM, :DocTestSetup, :(using HKQM); recursive=true)

makedocs(
    sitename = "HKQM",
    format = Documenter.HTML(),
    modules = [HKQM]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
