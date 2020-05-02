
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "1-Particle Basis"  begin

##


using SHIPs, Random
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d


##

@info("Basic test of PIBasis construction and evaluation")
maxdeg = 10
r0 = 1.0
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = SHIPs.SparsePSHDegree()
P1 = SHIPs.BasicPSH1pBasis(Pr; species = :X, D = D)

basis = SHIPs.PIBasis(P1, 2, D, maxdeg)

# check single-species
Nat = 15
Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, :X)
AA = evaluate(basis, Rs, Zs, z0)
println(@test(length(basis, z0) == length(AA)))

# check multi-species
maxdeg = 5
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
species = [:C, :O, :H]
P1 = SHIPs.BasicPSH1pBasis(Pr; species = [:C, :O, :H], D = D)
basis = SHIPs.PIBasis(P1, 3, D, maxdeg)
Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
AA = evaluate(basis, Rs, Zs, z0)
println(@test(length(basis, z0) == length(AA)))

##

@info("Check permutation invariance")
for species in (:X, :Si, [:C, :O, :H])
   Nat = 15
   P1 = SHIPs.BasicPSH1pBasis(Pr; species = species)
   basis = SHIPs.PIBasis(P1, 3, D, maxdeg)
   for ntest = 1:10
      Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
      p = randperm(length(Rs))
      print_tf(@test(evaluate(basis, Rs, Zs, z0) ≈
                     evaluate(basis, Rs[p], Zs[p], z0)))
   end
end

##

end
