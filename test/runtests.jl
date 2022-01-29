

using ACE, Test, Printf, LinearAlgebra, StaticArrays, BenchmarkTools

##
@testset "ACE.jl" begin
    # ------------------------------------------
    #   basic polynomial basis building blocks
    @testset "Ylm" begin include("polynomials/test_ylm.jl") end
    @testset "TestWigner" begin include("testing/test_wigner.jl") end
    @testset "Transforms" begin include("polynomials/test_transforms.jl") end
    @testset "OrthogonalPolynomials" begin include("polynomials/test_orthpolys.jl") end

    # ------------------------------------------
    #  states, .. 
    @testset "States" begin include("test_states.jl") end 

    # --------------------------------------------
    # core permutation-invariant functionality
    @testset "1-Particle Basis"  begin include("test_1pbasis.jl") end
    @testset "MultipleFeatures" begin  include("test_multigrads.jl") end
    @testset "Categorical1pBasis" begin include("test_discrete.jl") end
    @testset "PIBasis" begin include("test_pibasis.jl") end

    # ------------------------
    #   O(3) equi-variance
    # @testset "Clebsch-Gordan" begin include("test_cg.jl") end
    @testset "SymmetricBasis" begin include("test_symmbasis.jl") end
    @testset "EuclideanVector" begin include("test_euclvec.jl") end
    @testset "EuclideanMatrix" begin include("test_EuclideanMatrix.jl") end
    @testset "Multiple SH Bases" begin include("test_multish.jl") end

    # Model tests
    @testset "LinearACEModel"  begin include("test_linearmodel.jl") end
    @testset "MultipleProperties"  begin include("test_multiprop.jl") end
    @testset "AD-LinearACEModel"  begin include("test_admodel.jl") end 

    # Experimental material
    @testset "Multipliers" begin include("test_multiplier.jl") end
    @testset "Bonds" begin include("test_bonds.jl") end
    @testset "Bond BasisSelectors" begin include("test_bondbasisselectors.jl") end
end


    # -----------------------------------------
    #    old tests to be re-introduced - maybe
    # include("test_real.jl")
    # include("test_orth.jl")
    # include("bonds/test_cylindrical.jl")
    # include("bonds/test_fourier.jl")
    # include("bonds/test_envpairbasis.jl")
