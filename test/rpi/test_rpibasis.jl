
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


@testset "RPIBasis"  begin

#---


using ACE
using Random, Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d, evaluate_ed
using JuLIP.MLIPs: combine


#---

@info("Basic test of RPIBasis construction and evaluation")
maxdeg = 15
N = 3
r0 = 1.0
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = SparsePSHDegree()
P1 = BasicPSH1pBasis(Pr; species = :X, D = D)

#---

pibasis = PIBasis(P1, N, D, maxdeg)
rpibasis = RPIBasis(P1, N, D, maxdeg)

#---
@info("Basis construction and evaluation checks")
@info("check single species")
Nat = 15
Rs, Zs, z0 = rand_nhd(Nat, Pr, :X)
B = evaluate(rpibasis, Rs, Zs, z0)
println(@test(length(rpibasis) == length(B)))
dB = evaluate_d(rpibasis, Rs, Zs, z0)
println(@test(size(dB) == (length(rpibasis), length(Rs))))
B_, dB_ = evaluate_ed(rpibasis, Rs, Zs, z0)
println(@test (B_ ≈ B) && (dB_ ≈ dB))

#---
@info("check multi-species")
maxdeg = 5
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
species = [:C, :O, :H]
P1 = ACE.BasicPSH1pBasis(Pr; species = species, D = D)
basis = ACE.RPIBasis(P1, N, D, maxdeg)
Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
B = evaluate(basis, Rs, Zs, z0)
println(@test(length(basis) == length(B)))
dB = evaluate_d(basis, Rs, Zs, z0)
println(@test(size(dB) == (length(basis), length(Rs))))
B_, dB_ = evaluate_ed(basis, Rs, Zs, z0)
println(@test (B_ ≈ B) && (dB_ ≈ dB))

#---

degrees = [ 12, 10, 8, 8, 8, 8 ]

@info("Check a few basis properties ")
# for species in (:X, :Si) # , [:C, :O, :H])
for species in (:X, :Si, [:C, :O, :H]), N = 1:length(degrees)
   local Rs, Zs, z0, B, dB, basis, D, P1, Nat
   Nat = 15
   D = SparsePSHDegree()
   P1 = ACE.BasicPSH1pBasis(Pr; species = species)
   basis = ACE.RPIBasis(P1, N, D, degrees[N])
   @info("species = $species; N = $N; deg = $(degrees[N]); len = $(length(basis))")
   @info("   check (de-)serialization")
   println(@test(all(JuLIP.Testing.test_fio(basis))))
   @info("   isometry and permutation invariance")
   for ntest = 1:30
      Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
      Rsp, Zsp = ACE.rand_sym(Rs, Zs)
      print_tf(@test(evaluate(basis, Rs, Zs, z0) ≈
                     evaluate(basis, Rsp, Zsp, z0)))
   end
   println()
   @info("   check derivatives")
   for ntest = 1:30
      Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
      B = evaluate(basis, Rs, Zs, z0)
      dB = evaluate_d(basis, Rs, Zs, z0)
      Us = [ rand(eltype(Rs)) .- 0.5 for _=1:length(Rs) ]
      dB_dUs = transpose.(dB) * Us
      errs = []
      for p = 2:12
         h = 0.1^p
         B_h = evaluate(basis, Rs + h * Us, Zs, z0)
         dB_h = (B_h - B) / h
         # @show norm(dAA_h - dAA_dUs, Inf)
         push!(errs, norm(dB_h - dB_dUs, Inf))
      end
      success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
      print_tf(@test success)
   end
   println()
   @info("   check combine")
   coeffs = randcoeffs(basis)
   V = combine(basis, coeffs)
   Vst = standardevaluator(V)
   for ntest = 1:30
      Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
      v = evaluate(V, Rs, Zs, z0)
      vst = evaluate(Vst, Rs, Zs, z0)
      cdotB = dot(coeffs, evaluate(basis, Rs, Zs, z0))
      print_tf(@test v ≈ cdotB ≈ vst)
   end
   println()
   @info("   check graph evaluator")
   basisst = standardevaluator(basis)
   for ntest = 1:30
      env = ACE.rand_nhd(Nat, Pr, species)
      print_tf(@test evaluate(basisst, env...) ≈ evaluate(basis, env...))
      print_tf(@test evaluate_d(basisst, env...) ≈ evaluate_d(basis, env...))
   end
   println()
end


#---

end
