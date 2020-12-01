

@testset "PolyProducts" begin

@info("--------- Testing Products of Polynomials ----------")

##
using ACE, Test, ForwardDiff, JuLIP, JuLIP.Testing

using LinearAlgebra: norm, cond
using ACE.OrthPolys: OrthPolyBasis
using JuLIP: evaluate, evaluate_d

##

maxn = 30
tdf = rand(200)
ww = 1.0 .+ rand(200)
Jd = OrthPolyBasis(maxn, 2, 1.0, 1, -1.0, tdf, ww)

##

@info("Testing the radial products")
coeffs = ACE.OrthPolys.OrthPolyProdCoeffs(Jd)

for n1 = 1:5, n2 = 1:5
   # product
   f1(x) = evaluate(Jd, x)[n1] * evaluate(Jd, x)[n2]
   # expansion
   f2(x) = ( P = coeffs(n1, n2);
             sum( evaluate(Jd, x)[ν] * P[ν]  for ν = 1:length(P) ) )
   for ntest = 1:10
      x = 2 * rand() - 1
      print_tf(@test f1(x) ≈ f2(x))
   end
end

##
end
