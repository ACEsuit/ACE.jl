
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions
# --------------------------------------------------------------------------


using Documenter, ACE

makedocs(sitename="ACE.jl Documentation",
         pages = [
        "Home" => "index.md",
        "Introduction" => "intro.md",
        "Getting Started" => "gettingstarted.md",
        "Developer Docs" => "devel.md",
        "ED-Bonds" => "envpairbasis.md"
        # "Subsection" => [
        #     ...
        # ]
        ])

# deploydocs(
#     repo = "github.com/JuliaMolSim/ACE.jl.git",
# )
