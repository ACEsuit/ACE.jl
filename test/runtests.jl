

using ACE, Test, Printf, LinearAlgebra, StaticArrays, BenchmarkTools

##
@testset "ACE.jl" begin
    # ------------------------------------------
    #   basic polynomial basis building blocks
    @testset "Ylm" begin include("polynomials/test_ylm.jl") end 
    @testset "TestWigner" begin include("testing/test_wigner.jl") end 
    @testset "Transforms" begin include("polynomials/test_transforms.jl") end 
    @testset "OrthogonalPolynomials" begin include("polynomials/test_orthpolys.jl") end 

    # --------------------------------------------
    # core permutation-invariant functionality
    @testset "1-Particle Basis"  begin include("test_1pbasis.jl") end 
    @testset "MultipleFeatures" begin  include("test_multigrads.jl") end 
    @testset "PIBasis" begin include("test_pibasis.jl") end

    # ------------------------
    #   O(3) equi-variance
    @testset "Clebsch-Gordan" begin include("test_cg.jl") end
    @testset "SymmetricBasis" begin include("test_symmbasis.jl") end
    @testset "EuclideanVector" begin include("test_euclvec.jl") end

    # Model tests 
    @testset "LinearACEModel"  begin include("test_linearmodel.jl") end 
    @testset "MultipleProperties"  begin include("test_multiprop.jl") end 

end


    # -----------------------------------------
    #    old tests to be re-introduced - maybe
    # include("test_real.jl")
    # include("test_orth.jl")
    # include("bonds/test_cylindrical.jl")
    # include("bonds/test_fourier.jl")
    # include("bonds/test_envpairbasis.jl")
