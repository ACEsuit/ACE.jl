
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


@testset "PIBasis"  begin

#---


using ACE, Random
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d


#---

@info("Basic test of PIBasis construction and evaluation")

ord = 5
maxdeg = 10
r0 = 1.0
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = ACE.SparsePSHDegree()
P1 = ACE.BasicPSH1pBasis(Pr; species = :X, D = D)

dagbasis = ACE.PIBasis(P1, ord, D, maxdeg)
basis = standardevaluator(dagbasis)

# check single-species
Nat = 15
Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, :X)
AA = evaluate(basis, Rs, Zs, z0)
println(@test(length(basis) == length(AA)))

# construct a basis with dag-evaluator, and check they are identical!!!
AAdag = evaluate(dagbasis, Rs, Zs, z0)
println(@test(AA ≈ AAdag))

dAA = evaluate_d(basis, Rs, Zs, z0
   )
dAAdag = evaluate_d(dagbasis, Rs, Zs, z0)
println(@test dAA ≈ dAAdag)

println(@test all(JuLIP.Testing.test_fio(dagbasis)))

#--- check multi-species
maxdeg = 5
ord = 3
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
species = [:C, :O, :H]
P1 = ACE.BasicPSH1pBasis(Pr; species = [:C, :O, :H], D = D)
dagbasis = ACE.PIBasis(P1, ord, D, maxdeg)
basis = standardevaluator(dagbasis)
Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
AA = evaluate(basis, Rs, Zs, z0)
println(@test(length(basis) == length(AA)))
dAA = evaluate_d(basis, Rs, Zs, z0)
println(@test(length(basis) == length(AA)))

# construct a basis with dag-evaluator, and check they are identical!!!
AAdag = evaluate(dagbasis, Rs, Zs, z0)
println(@test(AA ≈ AAdag))

dAA = evaluate_d(basis, Rs, Zs, z0)
dAAdag = evaluate_d(dagbasis, Rs, Zs, z0)
println(@test dAA ≈ dAAdag)

println(@test all(JuLIP.Testing.test_fio(dagbasis)))
#---

@info("Check several properties of PIBasis")
for species in (:X, :Si, [:C, :O, :H]), N = 1:5
   local AA, AAdag, dAA, dAAdag, dagbasis, basis, Rs, Zs, z0
   maxdeg = 7
   Nat = 15
   P1 = ACE.BasicPSH1pBasis(Pr; species = species)
   dagbasis = ACE.PIBasis(P1, N, D, maxdeg)
   basis = standardevaluator(dagbasis)
   @info("species = $species; N = $N; length = $(length(basis))")
   @info("test (de-)serialisation")
   # only require dagbasis to deserialize correctly
   println(@test all(JuLIP.Testing.test_fio(dagbasis)))
   @info("Check Permutation invariance")
   for ntest = 1:20
      Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
      p = randperm(length(Rs))
      print_tf(@test(evaluate(basis, Rs, Zs, z0) ≈
                     evaluate(basis, Rs[p], Zs[p], z0)))
   end
   println()
   @info("Check gradients")
   for ntest = 1:20
      Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
      AA = evaluate(basis, Rs, Zs, z0)
      dAA = evaluate_d(basis, Rs, Zs, z0)
      Us = [ rand(eltype(Rs)) .- 0.5 for _=1:length(Rs) ]
      dAA_dUs = transpose.(dAA) * Us
      errs = []
      for p = 2:12
         h = 0.1^p
         AA_h = evaluate(basis, Rs + h * Us, Zs, z0)
         dAA_h = (AA_h - AA) / h
         # @show norm(dAA_h - dAA_dUs, Inf)
         push!(errs, norm(dAA_h - dAA_dUs, Inf))
      end
      success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
      print_tf(@test success)
   end

   println()
   @info("Check Standard=DAG Evaluator")
   for ntest = 1:20
      Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
      AA = evaluate(basis, Rs, Zs, z0)
      AAdag = evaluate(dagbasis, Rs, Zs, z0)
      print_tf(@test AA ≈ AAdag)

      dAA = evaluate_d(basis, Rs, Zs, z0)
      dAAdag = evaluate_d(dagbasis, Rs, Zs, z0)
      print_tf(@test dAA ≈ dAAdag)
   end
   println()
end
println()

#---


end
