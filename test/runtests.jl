
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

using PoSH
using Test, Printf, LinearAlgebra, StaticArrays, BenchmarkTools, Test
using JuLIP, JuLIP.Testing
using JuLIP.Potentials: evaluate!, evaluate_d!, evaluate, evaluate_d
using PoSH: PolyTransform, PolyCutoff1s, PolyCutoff2s, eval_basis

##

@testset "PoSH.jl" begin
    include("pairpots/test_jacobi.jl")
    include("pairpots/test_transforms.jl")
    include("pairpots/test_basics.jl")
    include("pairpots/test_repulsion.jl")
    # ----------------------
    include("test_ylm.jl")
    include("test_cg.jl")
    # ----------------------
    include("test_basis.jl")
    include("test_filter_rpi_alg.jl")
    include("test_fast.jl")
    include("test_multispecies.jl")
    include("test_orth.jl")
    include("test_descriptor.jl")
end

# TODO: this test needs to be rewritten!
#       but maybe it is no longer needed? The rotation-invariance tests
#       somehow take care of it???
# include("test_Bcoeffs.jl")
