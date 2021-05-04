
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------

using ACE
using Test, Printf, LinearAlgebra, StaticArrays, BenchmarkTools, Test
using JuLIP, JuLIP.Testing
using JuLIP: evaluate!, evaluate_d!, evaluate, evaluate_d
using ACE: PolyTransform, PolyCutoff1s, PolyCutoff2s

##

@testset "ACE.jl" begin
    include("pairpots/test_jacobi.jl")
    include("pairpots/test_transforms.jl")
    include("pairpots/test_basics.jl")
    include("pairpots/test_repulsion.jl")
    include("pairpots/test_orthpolys.jl")
    # ----------------------
    include("test_ylm.jl")
    include("test_rylm.jl")
    include("test_cg.jl")
    # ----------------------
    include("test_basis.jl")
    include("test_filter_rpi_alg.jl")
    include("test_fast.jl")
    include("test_real.jl")
    include("test_multispecies.jl")
    include("test_orth.jl")
    # include("test_descriptor.jl")
    # ----------------------
    include("test_compressA.jl")
    # ---------------------- bonds stuff
    include("bonds/test_cylindrical.jl")
    include("bonds/test_fourier.jl")
    include("bonds/test_envpairbasis.jl")
end
