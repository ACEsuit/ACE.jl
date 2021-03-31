


@testset "TestWigner" begin

##
using ACE, StaticArrays
using ACEbase

##

@info("Check correctness of Wigner-D matrices")

L = 1
φ = ACE.SphericalVector(L; T = ComplexF64)
SH = ACE.SphericalHarmonics.SHBasis(1)

for ntest = 1:20
   Q, D = ACE.Wigner.rand_QD(φ)
   x = randn(SVector{3, Float64})
   x = x / norm(x)
   Y1 = evaluate(SH, x)[2:end]
   D_Y1_Q = D' * evaluate(SH, Q * x)[2:end]
   println((@test(Y1 ≈ D_Y1_Q)))
end
println()

##
end
