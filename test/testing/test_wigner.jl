


##
using ACE, StaticArrays, ACE.SphericalHarmonics;
using ACEbase
using ACE.SphericalHarmonics: index_y;
using ACE: evaluate
using Random, Printf, Test, ACE.Testing
using ACE, Test, Printf, LinearAlgebra, BenchmarkTools


##

for L = 0:5
   @info("Check correctness of Wigner-D matrix of order $L")
   SH = ACE.SphericalHarmonics.SHBasis(L)
   for ntest = 1:20
      Q, D = ACE.Wigner.rand_QD(L)
      x = randn(SVector{3, Float64})
      x = x / norm(x)
      Y1 = evaluate(SH, x)[(L^2+1):(L+1)^2]
      Dt_Y1_Q = D' * evaluate(SH, Q * x)[(L^2+1):(L+1)^2]
      print_tf(@test isapprox(Y1, Dt_Y1_Q, rtol=1e-10))
   end
   println()
end

##
