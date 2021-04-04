
# --------------------------------------------------------------------------
# ACE.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


#@testset "SymmetricBasis"  begin

#---

using ACE
using Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: _b2llnn, evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis, coupling_coeffs, Rot3DCoeffsEquiv,get_spec, rfltype
using ACE.Random: rand_rot, rand_refl


# construct the 1p-basis
D = NaiveTotalDegree()
maxdeg = 3
ord = 3

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

# generate a configuration
nX = 10
X0 = rand(EuclideanVectorState, B1p.bases[1])
Xs = rand(EuclideanVectorState, B1p.bases[1], nX)

#---

@info("SymmetricBasis construction and evaluation: Invariant Scalar")

φ = ACE.EuclideanVector()
pibasis = PIBasis(B1p, ord, maxdeg; property = φ, isreal=false)
basis = SymmetricBasis(pibasis, φ)

BB = evaluate(basis, Xs, X0)

# a stupid but necessary test
BB1 = basis.A2Bmap * evaluate(basis.pibasis, Xs, X0)
println(@test isapprox(BB, BB1, rtol=1e-10))


"""Test set for equivariance properties"""

tol = 1E-9


begin
      @info("check for rotation equivariance")
      using ReferenceFrameRotations: angle_to_dcm
      Q = angle_to_dcm(.5,.4,-.3, :ZYX)

      X0_rot = Q * X0
      Xs_rot = [Q * X for X in Xs]
      BB_rot = evaluate(basis, Xs_rot, X0_rot)

      @test all([ norm(Q*BB[i]-BB_rot[i])<tol for i in 1:length(BB)])
end


begin
      @info("check for inversion equivariance")

      BB_inv = evaluate(basis, -Xs, -X0)

      @test all([ norm(-BB[i]-BB_inv[i])<tol for i in 1:length(BB)])
end



begin
      @info("check for permutation equivariance")
      using Combinatorics: permutations

      σ = randperm(nX)
      BB_perm = evaluate(basis, Xs[σ], X0)
      @test all([ norm(BB[i]-BB_perm[i])<tol for i in 1:length(BB)])
end
