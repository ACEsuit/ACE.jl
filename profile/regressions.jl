
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using BenchmarkTools
globalsuite = BenchmarkGroup()

include("profile_ylm.jl")

# judge(minimum(results), minimum(results1)) |> display
