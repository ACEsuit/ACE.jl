

using ACE, Test, Printf, LinearAlgebra, StaticArrays, BenchmarkTools

##
@testset "ACE.jl" begin
    # ------------------------------------------
    #   basic polynomial basis building blocks
    include("polynomials/test_ylm.jl")
    include("testing/test_wigner.jl")
    include("polynomials/test_transforms.jl")
    include("polynomials/test_orthpolys.jl")

    # --------------------------------------------
    # core permutation-invariant functionality
    include("test_scal1pbasis.jl")
    include("test_1pbasis.jl")
    include("test_pibasis.jl")
    # include("test_pipot.jl")   # TODO...

    # ------------------------
    #   rotation_invariance
    include("test_cg.jl")
    include("test_symmbasis.jl")
    include("test_euclvec.jl")

    # # ----------------------
    # #   miscallaneous ...
    # include("compat/test_compat_v05.jl")
    # include("compat/test_compat.jl")
    # include("test_any.jl")

    # ----------------------------------
    #    old tests to be re-introduced
    # include("test_real.jl")
    # include("test_orth.jl")
    # include("test_descriptor.jl")
    # include("bonds/test_cylindrical.jl")
    # include("bonds/test_fourier.jl")
    # include("bonds/test_envpairbasis.jl")
end
