
# --------------------------------------------------------------------------
# ACE.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@testset "PIBasis"  begin

#---


using ACE, Random
using Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, NaiveTotalDegree


#---

@info("Basic test of PIBasis construction and evaluation")

D = NaiveTotalDegree()
maxdeg = 6
ord = 3
φ = ACE.Invariant()

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

pibasis = PIBasis(B1p, ord, maxdeg; property = φ)

# generate a configuration
nX = 10
Xs = rand(EuclideanVectorState, B1p.bases[1], nX)
cfg = ACEConfig(Xs)

AA = evaluate(pibasis, cfg)

println(@test(length(pibasis) == length(AA)))

spec = ACE.get_spec(pibasis)
println(@test all(length(b) > 0 for b in spec))

spec_naive = [
    [ ACE.get_spec(B1p, pibasis.spec.iAA2iA[iAA, t])
      for t = 1:pibasis.spec.orders[iAA] ]   for iAA = 1:length(pibasis)
    ]

println(@test spec == spec_naive)


# get inverse Aspec
inv_spec1 = Dict{Any, Int}()
for (i, b1) in enumerate(ACE.get_spec(B1p))
  inv_spec1[b1] = i
end

# a really naive implementation of PIBasis to check correctness
A = evaluate(B1p, cfg)
AA_naive =  [
      real(prod( A[ inv_spec1[ b1 ] ] for b1 in b )) for b in spec ]
println(@test( AA_naive ≈ AA ))


# TODO: test derivatives!!!

# dAA = evaluate_d(basis, Rs, Zs, z0
#    )
# dAAdag = evaluate_d(dagbasis, Rs, Zs, z0)
# println(@test dAA ≈ dAAdag)

# println(@test all(ACE.Testing.test_fio(pibasis))
#    )


# @info("Check several properties of PIBasis")
# for species in (:X, :Si, [:C, :O, :H]), N = 1:5
#    local AA, AAdag, dAA, dAAdag, dagbasis, basis, Rs, Zs, z0
#    maxdeg = 7
#    Nat = 15
#    P1 = ACE.RnYlm1pBasis(Pr; species = species)
#    dagbasis = ACE.PIBasis(P1, N, D, maxdeg)
#    basis = standardevaluator(dagbasis)
#    @info("species = $species; N = $N; length = $(length(basis))")
#    @info("test (de-)serialisation")
#    # only require dagbasis to deserialize correctly
#    println(@test all(JuLIP.Testing.test_fio(dagbasis)))
#    @info("Check Permutation invariance")
#    for ntest = 1:20
#       Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
#       p = randperm(length(Rs))
#       print_tf(@test(evaluate(basis, Rs, Zs, z0) ≈
#                      evaluate(basis, Rs[p], Zs[p], z0)))
#    end
#    println()
#    @info("Check gradients")
#    for ntest = 1:20
#       Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
#       AA = evaluate(basis, Rs, Zs, z0)
#       dAA = evaluate_d(basis, Rs, Zs, z0)
#       Us = [ rand(eltype(Rs)) .- 0.5 for _=1:length(Rs) ]
#       dAA_dUs = transpose.(dAA) * Us
#       errs = []
#       for p = 2:12
#          h = 0.1^p
#          AA_h = evaluate(basis, Rs + h * Us, Zs, z0)
#          dAA_h = (AA_h - AA) / h
#          # @show norm(dAA_h - dAA_dUs, Inf)
#          push!(errs, norm(dAA_h - dAA_dUs, Inf))
#       end
#       success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
#       print_tf(@test success)
#    end
#
#    println()
#    @info("Check Standard=DAG Evaluator")
#    for ntest = 1:20
#       Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
#       AA = evaluate(basis, Rs, Zs, z0)
#       AAdag = evaluate(dagbasis, Rs, Zs, z0)
#       print_tf(@test AA ≈ AAdag)
#
#       dAA = evaluate_d(basis, Rs, Zs, z0)
#       dAAdag = evaluate_d(dagbasis, Rs, Zs, z0)
#       print_tf(@test dAA ≈ dAAdag)
#    end
#    println()
# end
# println()

#---


end
