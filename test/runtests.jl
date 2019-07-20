
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

using SHIPs
using Test, Printf, LinearAlgebra, StaticArrays, BenchmarkTools, Test
using JuLIP, JuLIP.Testing
using JuLIP.Potentials: evaluate!, evaluate_d!, evaluate, evaluate_d
using SHIPs: PolyTransform, PolyCutoff1s, PolyCutoff2s, eval_basis

@testset "SHIPs.jl" begin
    include("test_jacobi.jl")
    include("test_transforms.jl")
    include("test_ylm.jl")
    include("test_cg.jl")
    include("test_Bcoeffs.jl")
    include("test_basis.jl")
    # include("test_fast.jl")
    # include("test_pair.jl")
end
