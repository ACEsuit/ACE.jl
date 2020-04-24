
using BenchmarkTools
globalsuite = BenchmarkGroup()

include("profile_ylm.jl")

# judge(minimum(results), minimum(results1)) |> display
