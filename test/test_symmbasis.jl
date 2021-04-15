
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
maxdeg = 6
ord = 3

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

# generate a configuration
nX = 10
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

## Testing derivatives

@info("  ... Derivatives")
tmpd = ACE.alloc_temp_d(basis, length(cfg))
dB = ACE.evaluate_d(basis, cfg)

for ntest = 1:30
   Us = randn(SVector{3, Float64}, length(Xs))
   c = randn(length(basis))
   F = t -> sum(c .* ACE.evaluate(basis, ACEConfig(Xs + t[1] * Us))).val
   dF = t -> [ Us' * sum(c .* ACE.evaluate_d(basis, ACEConfig(Xs + t[1] * Us)), dims=1)[:] ]
   print_tf(@test fdtest(F, dF, [0.0], verbose=false))
end


#---
@info("SymmetricBasis construction and evaluation: Spherical Vector")

__L2syms = [:s, :p, :d, :f, :g, :h, :i, :k]
__syms2L = Dict( [sym => L-1 for (L, sym) in enumerate(__L2syms)]... )
get_orbsym(L::Integer)  = __L2syms[L+1]

for L = 0 : 3

      @info "Tests for L = $L ⇿ $(get_orbsym(0))-$(get_orbsym(L)) block"

      φ = ACE.SphericalVector(L; T = ComplexF64)
      pibasis = PIBasis(B1p, ord, maxdeg; property = φ, isreal = false)
      basis = SymmetricBasis(pibasis, φ)
      BB = evaluate(basis, cfg)

      for ntest = 1:10
            Q, D = ACE.Wigner.rand_QD(L)
            cfg1 = ACEConfig( shuffle(Ref(Q) .* Xs) )
            BB1 = evaluate(basis, cfg1)
            DtxBB1 = Ref(D') .* BB1
            print_tf(@test isapprox(DtxBB1, BB, rtol=1e-10))
      end
      println()
end

#---
@info("SymmetricBasis construction and evaluation: Spherical Matrix")

for L1 = 0:3
   for L2 = 0:3

      @info "Tests for L₁ = $L1, L₂ = $L2 ⇿ $(get_orbsym(L1))-$(get_orbsym(L2)) block"

      φ = ACE.SphericalMatrix(L1, L2; T = ComplexF64)
      pibasis = PIBasis(B1p, ord, maxdeg; property = φ, isreal = false)
      basis = SymmetricBasis(pibasis, φ)
      BB = evaluate(basis, cfg)

      for ntest = 1:10
         Q, D1, D2 = ACE.Wigner.rand_QD(L1, L2)
         cfg1 = ACEConfig( shuffle(Ref(Q) .* Xs) )
         BB1 = evaluate(basis, cfg1)
         D1txBB1xD2 = Ref(D1') .* BB1 .* Ref(D2)
         print_tf(@test isapprox(D1txBB1xD2, BB, rtol=1e-10))
      end
      println()
   end
end

#---
@info("Consistency between SphericalVector & SphericalMatrix")

for L = 0:3

   @info "L = $L"

   φ1 = ACE.SphericalVector(L; T = ComplexF64)
   pibasis1 = PIBasis(B1p, ord, maxdeg; property = φ1, isreal = false)
   basis1 = SymmetricBasis(pibasis1, φ1)
   φ2 = ACE.SphericalMatrix(L, 0; T = ComplexF64)
   pibasis2 = PIBasis(B1p, ord, maxdeg; property = φ2, isreal = false)
   basis2 = SymmetricBasis(pibasis2, φ2)

   for ntest = 1:10

      Xs = rand(EuclideanVectorState, B1p.bases[1], nX)
      cfg = ACEConfig(Xs)

      BBvec = evaluate(basis1, cfg)
      value1 = [reshape(BBvec[i].val, 2L+1, 1) for i in 1:length(BBvec)]


      BBmat = evaluate(basis2, cfg)
      value2 = [BBmat[i].val for i in 1:length(BBvec)]

      print_tf(@test isapprox(value1, value2, rtol=1e-10))

   end
   println()

end

#---
@info("Consistency between Invariant Scalar & SphericalMatrix")

φ = ACE.Invariant()
pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
basis = SymmetricBasis(pibasis, φ)
φ2 = ACE.SphericalMatrix(0, 0; T = ComplexF64)
pibasis2 = PIBasis(B1p, ord, maxdeg; property = φ2, isreal = false)
basis2 = SymmetricBasis(pibasis2, φ2)

for ntest = 1:10

   Xs = rand(EuclideanVectorState, B1p.bases[1], nX)
   cfg = ACEConfig(Xs)

   BB = evaluate(basis, cfg)
   BBsca = [BB[i].val for i in 1:length(BB)]

   BB2 =  evaluate(basis2, cfg)
   BBCFlo = [ComplexF64(BB2[i].val...) for i in 1:length(BB2)]

   print_tf(@test isapprox(BBsca, BBCFlo, rtol=1e-10))

end
println()





#---
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
