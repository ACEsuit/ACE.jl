
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs, Test, Printf, LinearAlgebra, StaticArrays, BenchmarkTools,
      JuLIP, JuLIP.Testing

##
@testset "SHIPs.jl" begin
    # ------------------------------------------
    #   basic polynomial basis building blocks
    include("polynomials/test_ylm.jl")
    include("polynomials/test_rylm.jl")
    include("polynomials/test_transforms.jl")
    include("polynomials/test_orthpolys.jl")

    # --------------------------------------------
    # core permutation-invariant functionality
    include("test_1pbasis.jl")
    include("test_pibasis.jl")
    include("test_pipot.jl")

    # ------------------------
    #   rotation_invariance
    include("rpi/test_cg.jl")
    include("rpi/test_rpibasis.jl")

    # ----------------------
    #   pair potentials
    include("pair/test_pair_basis.jl")
    include("pair/test_pair_pot.jl")
    include("pair/test_repulsion.jl")


    # ----------------------
    #   miscallaneous ...
    include("test_compat.jl")
    include("experimental/test_treeeval.jl")

    # include("test_real.jl")
    # include("test_orth.jl")
    # include("test_descriptor.jl")

    # ---------------------------
    #    env-dependent bonds
    # include("bonds/test_cylindrical.jl")
    # include("bonds/test_fourier.jl")
    # include("bonds/test_envpairbasis.jl")
end
