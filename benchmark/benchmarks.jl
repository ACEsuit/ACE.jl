
using BenchmarkTools

const SUITE = BenchmarkGroup()

include("bm_basis.jl")
SUITE["basis"] = basis_suite