using Documenter
using HKQM

DocMeta.setdocmeta!(HKQM, :DocTestSetup, :(using HKQM); recursive=true)

makedocs(
    sitename = "HKQM",
    format = Documenter.HTML(),
    modules = [HKQM],
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/tjjarvinen/HKQM.jl.git"
)
