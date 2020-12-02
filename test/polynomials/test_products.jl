

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

@info("Testing the Expansion of products of radial polynomials")
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

@info("Testing Expansion of Products of Spherical Harmonics")
using ACE.Orth: SHProdCoeffs
using ACE.SphericalHarmonics: SHBasis, index_y

coeffs = SHProdCoeffs()
for l1 = 0:4, l2=0:4, m1 = -l1:l1, m2 = -l2:l2
   f1 = let SH = SHBasis(10)
      x -> ( Y = evaluate(SH, x);
             Y[index_y(l1, m1)] * Y[index_y(l2, m2)] )
   end
   f2 = let P = coeffs(l1, m1, l2, m2), SH = SHBasis(10)
      x -> sum( p * evaluate(SH, x)[index_y(L, M)]
                for (L, M, p) in P )
   end
   nfail = 0
   for ntest = 1:10
      x = randn(JVec); x /= norm(x)
      nfail += (abs(f1(x) - f2(x)) > 1e-12)
   end
   print_tf(@test nfail == 0)
end

##




end
