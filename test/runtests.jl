
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

using SHIPs
using Test, Printf, LinearAlgebra, StaticArrays, BenchmarkTools, Test

print_tf(::Test.Pass) = printstyled("+", bold=true, color=:green)
print_tf(::Test.Fail) = printstyled("-", bold=true, color=:red)

@testset "SHIPs.jl" begin
    include("test_jacobi.jl")
    # include("test_transforms.jl")
    # include("test_ylm.jl")
    # include("test_cg.jl")
    # include("test_Bcoeffs.jl")
    # include("test_basis.jl")
end
