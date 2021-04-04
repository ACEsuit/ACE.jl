

@testset "SymmetricBasis"  begin

#---


using ACE
using Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis
using ACE.Random: rand_rot, rand_refl


# construct the 1p-basis
D = NaiveTotalDegree()
maxdeg = 6
ord = 3

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

# generate a configuration
nX = 10
X0 = rand(EuclideanVectorState, B1p.bases[1])
Xs = rand(EuclideanVectorState, B1p.bases[1], nX)

#---

@info("SymmetricBasis construction and evaluation: Euclidean Vector")

φ = ACE.EuclideanVector()
pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
basis = SymmetricBasis(pibasis, φ)

BB = evaluate(basis, Xs, X0)

# a stupid but necessary test
BB1 = basis.A2Bmap * evaluate(basis.pibasis, Xs, X0)
println(@test isapprox(BB, BB1, rtol=1e-10))

for ntest = 1:30
      Xs1 = shuffle(rand_refl(rand_rot(Xs)))
      BB1 = evaluate(basis, Xs1, X0)
      print_tf(@test isapprox(BB, BB1, rtol=1e-10))
end



end
