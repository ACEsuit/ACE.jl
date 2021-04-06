
@testset "SymmetricBasis"  begin

#---


using ACE
using StaticArrays, Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis
using ACE.Random: rand_rot, rand_refl

# Extra using Wigner for computing Wigner Matrix
using ACE.Wigner


# construct the 1p-basis
D = NaiveTotalDegree()
maxdeg = 3
ord = 1

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

# generate a configuration
nX = 1
Xs = rand(EuclideanVectorState, B1p.bases[1], nX)
cfg = ACEConfig(Xs)

#---

@info("SymmetricBasis construction and evaluation: Invariant Scalar")

φ = ACE.Invariant()
pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
basis = SymmetricBasis(pibasis, φ)

BB = evaluate(basis, cfg)

# a stupid but necessary test
BB1 = basis.A2Bmap * evaluate(basis.pibasis, cfg)
println(@test isapprox(BB, BB1, rtol=1e-10))

for ntest = 1:30
      Xs1 = shuffle(rand_refl(rand_rot(Xs)))
      BB1 = evaluate(basis, ACEConfig(Xs1))
      print_tf(@test isapprox(BB, BB1, rtol=1e-10))
end
println()


#---

L = 1
ord = 1
φ = ACE.SphericalVector(L; T = ComplexF64)
pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
basis = SymmetricBasis(pibasis, φ)
ACE.fltype(basis) == typeof(φ)

Xs = rand(EuclideanVectorState, B1p.bases[1], 1)
cfg = ACEConfig(Xs)
BB = evaluate(basis, cfg)
# for ntest = 1:30

Q, D = ACE.Wigner.rand_QD(φ)

SH = ACE.SphericalHarmonics.SHBasis(2)
x = randn(SVector{3, Float64})
#x = x / norm(x)
#Y0 = evaluate(SH, x)[1]
Y1 = evaluate(SH, x)[2:4]
#Y2 = evaluate(SH, x)[5:9]
#D_Y0_Q = D' * evaluate(SH, Q * x)[1]
D_Y1_Q = D' * evaluate(SH, Q * x)[2:4]
#D_Y2_Q = D' * evaluate(SH, Q * x)[5:9]
if Y1 ≈ D_Y1_Q
      println("Correct Wigner Matrix!")
end

Xs1 = Ref(Q) .* Xs
cfg1 = ACEConfig( Ref(Q) .* Xs )
BB1 = evaluate(basis, cfg1)
DxBB = Ref(D) .* BB
norm(BB1)
norm(DxBB)
#[norm(DxBB1[i]) for i in 1:9]
norm(BB1 - DxBB)


# end

##
# #---
# @info("Basis construction and evaluation checks")
# @info("check single species")
# Nat = 15
# Rs, Zs, z0 = rand_nhd(Nat, Pr, :X)
# B = evaluate(rpibasis, Rs, Zs, z0)
# println(@test(length(rpibasis) == length(B)))
# dB = evaluate_d(rpibasis, Rs, Zs, z0)
# println(@test(size(dB) == (length(rpibasis), length(Rs))))
# B_, dB_ = evaluate_ed(rpibasis, Rs, Zs, z0)
# println(@test (B_ ≈ B) && (dB_ ≈ dB))
#
# #---
# @info("check multi-species")
# maxdeg = 5
# Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
# species = [:C, :O, :H]
# P1 = ACE.RnYlm1pBasis(Pr; species = species, D = D)
# basis = ACE.RPIBasis(P1, N, D, maxdeg)
# Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
# B = evaluate(basis, Rs, Zs, z0)
# println(@test(length(basis) == length(B)))
# dB = evaluate_d(basis, Rs, Zs, z0)
# println(@test(size(dB) == (length(basis), length(Rs))))
# B_, dB_ = evaluate_ed(basis, Rs, Zs, z0)
# println(@test (B_ ≈ B) && (dB_ ≈ dB))
#
# #---
#
# degrees = [ 12, 10, 8, 8, 8, 8 ]
#
# @info("Check a few basis properties ")
# # for species in (:X, :Si) # , [:C, :O, :H])
# for species in (:X, :Si, [:C, :O, :H]), N = 1:length(degrees)
#    local Rs, Zs, z0, B, dB, basis, D, P1, Nat
#    Nat = 15
#    D = SparsePSHDegree()
#    P1 = ACE.RnYlm1pBasis(Pr; species = species)
#    basis = ACE.RPIBasis(P1, N, D, degrees[N])
#    @info("species = $species; N = $N; deg = $(degrees[N]); len = $(length(basis))")
#    @info("   check (de-)serialization")
#    println(@test(all(JuLIP.Testing.test_fio(basis))))
#    @info("   isometry and permutation invariance")
#    for ntest = 1:30
#       Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
#       Rsp, Zsp = ACE.rand_sym(Rs, Zs)
#       print_tf(@test(evaluate(basis, Rs, Zs, z0) ≈
#                      evaluate(basis, Rsp, Zsp, z0)))
#    end
#    println()
#    @info("   check derivatives")
#    for ntest = 1:30
#       Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
#       B = evaluate(basis, Rs, Zs, z0)
#       dB = evaluate_d(basis, Rs, Zs, z0)
#       Us = [ rand(eltype(Rs)) .- 0.5 for _=1:length(Rs) ]
#       dB_dUs = transpose.(dB) * Us
#       errs = []
#       for p = 2:12
#          h = 0.1^p
#          B_h = evaluate(basis, Rs + h * Us, Zs, z0)
#          dB_h = (B_h - B) / h
#          # @show norm(dAA_h - dAA_dUs, Inf)
#          push!(errs, norm(dB_h - dB_dUs, Inf))
#       end
#       success = (/(extrema(errs)...) < 1e-3) || (minimum(errs) < 1e-10)
#       print_tf(@test success)
#    end
#    println()
#    @info("   check combine")
#    coeffs = randcoeffs(basis)
#    V = combine(basis, coeffs)
#    Vst = standardevaluator(V)
#    for ntest = 1:30
#       Rs, Zs, z0 = ACE.rand_nhd(Nat, Pr, species)
#       v = evaluate(V, Rs, Zs, z0)
#       vst = evaluate(Vst, Rs, Zs, z0)
#       cdotB = dot(coeffs, evaluate(basis, Rs, Zs, z0))
#       print_tf(@test v ≈ cdotB ≈ vst)
#    end
#    println()
#    @info("   check graph evaluator")
#    basisst = standardevaluator(basis)
#    for ntest = 1:30
#       env = ACE.rand_nhd(Nat, Pr, species)
#       print_tf(@test evaluate(basisst, env...) ≈ evaluate(basis, env...))
#       print_tf(@test evaluate_d(basisst, env...) ≈ evaluate_d(basis, env...))
#    end
#    println()
# end
#

#---

end
