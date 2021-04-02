
# --------------------------------------------------------------------------
# ACE.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


#@testset "1-Particle Basis"  begin

##

using ACE
using Printf, Test, LinearAlgebra
using ACE: evaluate, evaluate_d, Rn1pBasis, Ylm1pBasis,
      EuclideanVectorState, Product1pBasis
using Random: shuffle

##

@info "Build a 1p basis from scratch"

maxdeg = 5
r0 = 1.0
rcut = 3.0

trans = PolyTransform(1, r0)
J = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
Rn = Rn1pBasis(J)
Ylm = Ylm1pBasis(maxdeg)
B1p = Product1pBasis( (Rn, Ylm) )
ACE.init1pspec!(B1p)

nX = 10
X0 = rand(EuclideanVectorState, Rn)
Xs = rand(EuclideanVectorState, Rn, nX)

A = evaluate(B1p, Xs, X0)
# evaluate_d(B1p, Xs, X0)

@info("test against manual summation")
A1 = sum( evaluate(B1p, X, X0) for X in Xs )
println(@test A1 ≈ A)

@info("test permutation invariance")
println(@test A ≈ evaluate(B1p, shuffle(Xs), X0))

# not sure what else we can suitably test here ...

#end
