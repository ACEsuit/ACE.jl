
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs, Test, Printf, LinearAlgebra, StaticArrays, BenchmarkTools,
      JuLIP, JuLIP.Testing

##

@testset "SHIPs.jl" begin
    # ----------------------
    include("polynomials/test_ylm.jl")
    include("polynomials/test_rylm.jl")
    include("polynomials/test_transforms.jl")
    include("polynomials/test_orthpolys.jl")
    # ----------------------
    include("test_1pbasis.jl")
    include("test_pibasis.jl")

    # ----------------------
    #   rotation_invariance
    include("rpi/test_cg.jl")
    include("rpi/test_rpibasis.jl") 

    # include("test_cg.jl")
    # include("pairpots/test_basics.jl")
    # include("pairpots/test_repulsion.jl")
    # include("test_basis.jl")
    # include("test_filter_rpi_alg.jl")
    # include("test_fast.jl")
    # include("test_real.jl")
    # include("test_multispecies.jl")
    # include("test_orth.jl")
    # include("test_descriptor.jl")
    # ----------------------
    # include("test_compressA.jl")
    # ---------------------- bonds stuff
    # include("bonds/test_cylindrical.jl")
    # include("bonds/test_fourier.jl")
    # include("bonds/test_envpairbasis.jl")
end

# *
