
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "RPIBasis"  begin

##


using SHIPs
using Random, Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d


##

@info("Basic test of RPIBasis construction and evaluation")
maxdeg = 4
N =  2
r0 = 1.0
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = SHIPs.SparsePSHDegree()
P1 = SHIPs.BasicPSH1pBasis(Pr; species = :X, D = D)

##

pibasis = SHIPs.PIBasis(P1, N, D, maxdeg)
rpibasis = SHIPs.RPIBasis(P1, N, D, maxdeg)


##
# check single-species
Nat = 15
Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, :X)
B = evaluate(rpibasis, Rs, Zs, z0)
println(@test(length(rpibasis) == length(B)))

# check multi-species
maxdeg = 5
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
species = [:C, :O, :H]
P1 = SHIPs.BasicPSH1pBasis(Pr; species = species, D = D)
basis = SHIPs.RPIBasis(P1, N, D, maxdeg)
Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
B = evaluate(basis, Rs, Zs, z0)
println(@test(length(basis) == length(B)))

##

degrees = [ 12, 10, 8, 8, 8, 8 ]

@info("Check isometry and permutation invariance")
# for species in (:X, :Si) # , [:C, :O, :H])
for species in (:X, :Si, [:C, :O, :H]), N = 1:length(degrees)
   @info("   species = $species; N = $N; degree = $(degrees[N])")
   Nat = 15
   D = SparsePSHDegree()
   P1 = SHIPs.BasicPSH1pBasis(Pr; species = species)
   basis = SHIPs.RPIBasis(P1, N, D, degrees[N])
   @info("   length(basis) = $(length(basis))")
   for ntest = 1:30
      Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
      Rsp, Zsp = SHIPs.rand_sym(Rs, Zs)
      print_tf(@test(evaluate(basis, Rs, Zs, z0) ≈
                     evaluate(basis, Rsp, Zsp, z0)))
   end
   println()
end

##

end
