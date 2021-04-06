


@testset "TestWigner"

begin

##
using ACE, StaticArrays, ACE.SphericalHarmonics;
using ACEbase
using ACE.SphericalHarmonics: index_y;
using ACE: evaluate
using LinearAlgebra
using Random, Printf, Test, ACE.Testing

##

@info("Check correctness of Wigner-D matrices")

L = 1
φ = ACE.SphericalVector(L; T = ComplexF64)
SH = ACE.SphericalHarmonics.SHBasis(2)

for ntest = 1:20
   Q, D = ACE.Wigner.rand_QD(φ)
   x = randn(SVector{3, Float64})
   x = x / norm(x)
   Y1 = evaluate(SH, x)[2:4]
   D_Y1_Q = D' * evaluate(SH, Q * x)[2:4]
   println(Y1 ≈ D_Y1_Q)
#   print_tf(@test isapprox(Y1, D_Y1_Q, rtol=1e-10))
end

##
end
