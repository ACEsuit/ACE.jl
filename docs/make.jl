
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using Documenter, SHIPs

makedocs(sitename="SHIPs.jl Documentation",
         pages = [
        "Home" => "index.md",
        "Introduction" => "intro.md",
        "Developer Docs" => "devel.md",
        "ED-Bonds" => "envpairbasis.md"
        # "Subsection" => [
        #     ...
        # ]
        ])

# deploydocs(
#     repo = "github.com/JuliaMolSim/SHIPs.jl.git",
# )
