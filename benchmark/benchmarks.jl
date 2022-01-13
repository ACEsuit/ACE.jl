
using BenchmarkTools

const SUITE = BenchmarkGroup()

include("bm_basis.jl")
include("bm_linear.jl")
SUITE["basis"] = basis_suite
SUITE["linear"] = linear_suite
