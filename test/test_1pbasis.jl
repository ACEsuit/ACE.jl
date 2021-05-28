
@testset "1-Particle Basis"  begin

##

using ACE
using Printf, Test, LinearAlgebra, StaticArrays
using ACE: evaluate, evaluate_d, Rn1pBasis, Ylm1pBasis,
      EuclideanVectorState, Product1pBasis
using Random: shuffle
using ACEbase.Testing: dirfdtest, fdtest, print_tf

##


@info "Build a 1p basis from scratch"

maxdeg = 5
r0 = 1.0
rcut = 3.0

trans = PolyTransform(1, r0)   # r -> x = 1/r^2
J = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)   #  J_n(x) * (x - xcut)^pcut
Rn = Rn1pBasis(J)
Ylm = Ylm1pBasis(maxdeg)
B1p = Product1pBasis( (Rn, Ylm) )
ACE.init1pspec!(B1p)

nX = 10
Xs = rand(EuclideanVectorState, Rn, nX)
cfg = ACEConfig(Xs)

A = evaluate(B1p, cfg)

@info("test against manual summation")
A1 = sum( evaluate(B1p, X) for X in Xs )
println(@test A1 ≈ A)

@info("test permutation invariance")
println(@test A ≈ evaluate(B1p, ACEConfig(shuffle(Xs))))

##

@info("Ylm1pBasis gradients")
Y = ACE.alloc_B(Ylm)
dY = ACE.alloc_dB(Ylm)
tmpd = ACE.alloc_temp_d(Ylm)
ACE.evaluate_ed!(Y, dY, tmpd, Ylm, Xs[1])

for ntest = 1:30
   x0 = randn(3)
   c = rand(length(Y))
   F = x -> sum(ACE.evaluate(Ylm, EuclideanVectorState(SVector{3}(x))) .* c)
   dF = x -> Vector(sum(ACE.evaluate_d(Ylm, EuclideanVectorState(SVector{3}(x))) .* c))
   print_tf(@test fdtest(F, dF, x0; verbose=false))
end
println()

##

@info("Rn1pBasis gradients")

for ntest = 1:30
   x0 = randn(3)
   c = rand(length(Rn))
   F = x -> sum(ACE.evaluate(Rn, EuclideanVectorState(SVector{3}(x))) .* c)
   dF = x -> Vector(sum(ACE.evaluate_d(Rn, EuclideanVectorState(SVector{3}(x))) .* c))
   print_tf(@test fdtest(F, dF, x0; verbose=false))
end
println()

##

@info("Product basis evaluate_ed! tests")

tmp_d = ACE.alloc_temp_d(B1p)
A1 = ACE.alloc_B(B1p)
A2 = ACE.alloc_B(B1p)
ACE.evaluate!(A1, tmp_d, B1p, cfg)
dA = ACE.alloc_dB(B1p, length(cfg))
ACE.evaluate_ed!(A2, dA, tmp_d, B1p, cfg)
println(@test A1 ≈ A2)

##
@info("Product basis gradient test")

for ntest = 1:30
   x0 = randn(3)
   c = rand(length(B1p))
   F = x -> sum(ACE.evaluate(B1p, EuclideanVectorState(SVector{3}(x))) .* c)
   dF = x -> Vector(sum(ACE.evaluate_ed(B1p, ACEConfig([EuclideanVectorState(SVector{3}(x))]))[2] .* c))
   F(x0)
   dF(x0)
   print_tf(@test fdtest(F, dF, x0; verbose=false))
end
println()
##

end
