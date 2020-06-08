
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "PIBasis"  begin

##


using SHIPs, Random
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d


##

@info("Basic test of PIBasis construction and evaluation")

ord = 5
maxdeg = 10
r0 = 1.0
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = SHIPs.SparsePSHDegree()
P1 = SHIPs.BasicPSH1pBasis(Pr; species = :X, D = D)

basis = SHIPs.PIBasis(P1, ord, D, maxdeg)

# check single-species
Nat = 15
Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, :X)
AA = evaluate(basis, Rs, Zs, z0)
println(@test(length(basis) == length(AA)))

# construct a basis with dag-evaluator, and check they are identical!!!
dagbasis = SHIPs.PIBasis(P1, ord, D, maxdeg, evaluator = :dag)
AAdag = evaluate(dagbasis, Rs, Zs, z0)
println(@test(AA ≈ AAdag))


## check multi-species
maxdeg = 5
ord = 3
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
species = [:C, :O, :H]
P1 = SHIPs.BasicPSH1pBasis(Pr; species = [:C, :O, :H], D = D)
basis = SHIPs.PIBasis(P1, ord, D, maxdeg)
Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
AA = evaluate(basis, Rs, Zs, z0)
println(@test(length(basis) == length(AA)))
dAA = evaluate_d(basis, Rs, Zs, z0)
println(@test(length(basis) == length(AA)))

# construct a basis with dag-evaluator, and check they are identical!!!
dagbasis = SHIPs.PIBasis(P1, ord, D, maxdeg, evaluator = :dag
   )
AAdag = evaluate(dagbasis, Rs, Zs, z0)
println(@test(AA ≈ AAdag))

##

@info("Check several properties of PIBasis")
for species in (:X, :Si, [:C, :O, :H]), N = 1:5
   local AA, AAdag, dAA
   maxdeg = 7
   Nat = 15
   P1 = SHIPs.BasicPSH1pBasis(Pr; species = species)
   basis = SHIPs.PIBasis(P1, N, D, maxdeg)
   @info("species = $species; N = $N; length = $(length(basis))")
   @info("test (de-)serialisation")
   println(@test all(JuLIP.Testing.test_fio(basis)))
   @info("Check Permutation invariance")
   for ntest = 1:20
      Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
      p = randperm(length(Rs))
      print_tf(@test(evaluate(basis, Rs, Zs, z0) ≈
                     evaluate(basis, Rs[p], Zs[p], z0)))
   end
   println()
   @info("Check gradients")
   for ntest = 1:20
      Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
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
   @info("Check Classic=DAG Evaluator")
   dagbasis = SHIPs.PIBasis(P1, N, D, maxdeg, evaluator = :dag)
   for ntest = 1:20
      Rs, Zs, z0 = SHIPs.rand_nhd(Nat, Pr, species)
      AA = evaluate(basis, Rs, Zs, z0)
      AAdag = evaluate(dagbasis, Rs, Zs, z0)
      print_tf(@test AA ≈ AAdag)
   end
   println()
end
println()

##


end
