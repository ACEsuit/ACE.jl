# some type piracy ...
# TODO: hack like this make #27 important!!!

using StaticArrays
import Base: *
*(a::SArray{Tuple{L1,L2,L3}}, b::SVector{L3}) where {L1, L2, L3} =
      reshape( reshape(a, L1*L2, L3) * b, L1, L2)


@testset "SymmetricBasis"  begin

#---

using ACE
using StaticArrays, Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis
using ACE.Random: rand_rot, rand_refl
using ACEbase.Testing: fdtest

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
# @time SymmetricBasis(pibasis, φ);

BB = evaluate(basis, cfg)

# a stupid but necessary test
AA = evaluate(basis.pibasis, cfg)
BB1 = basis.A2Bmap * AA
println(@test isapprox(BB, BB1, rtol=1e-10))

# check there are no superfluous columns
Iz = findall(iszero, sum(norm, basis.A2Bmap, dims=1)[:])
if !isempty(Iz)
   @warn("The A2B map for Invariants has $(length(Iz))/$(length(basis.pibasis)) zero-columns!!!!")
end

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
println()

#---
@info("SymmetricBasis construction and evaluation: Spherical Vector")

__L2syms = [:s, :p, :d, :f, :g, :h, :i, :k]
__syms2L = Dict( [sym => L-1 for (L, sym) in enumerate(__L2syms)]... )
get_orbsym(L::Integer)  = __L2syms[L+1]

for L = 0:3
   @info "Tests for L = $L ⇿ $(get_orbsym(0))-$(get_orbsym(L)) block"
   φ = ACE.SphericalVector(L; T = ComplexF64)
   pibasis = PIBasis(B1p, ord, maxdeg; property = φ, isreal = false)
   basis = SymmetricBasis(pibasis, φ)
   BB = evaluate(basis, cfg)

   Iz = findall(iszero, sum(norm, basis.A2Bmap, dims = 1))
   if !isempty(Iz)
      @warn("The A2B map for SphericalVector has $(length(Iz))/$(length(basis.pibasis)) zero-columns!!!!")
   end

   for ntest = 1:30
      Q, D = ACE.Wigner.rand_QD(L)
      cfg1 = ACEConfig( shuffle(Ref(Q) .* Xs) )
      BB1 = evaluate(basis, cfg1)
      DtxBB1 = Ref(D') .* BB1
      print_tf(@test isapprox(DtxBB1, BB, rtol=1e-10))
   end
   println()

   @info(" .... derivatives")
   for ntest = 1:30
      Us = randn(SVector{3, Float64}, length(Xs))
      C = randn(typeof(φ.val), length(basis))
      F = t -> sum( sum(c .* b.val)
                    for (c, b) in zip(C, ACE.evaluate(basis, ACEConfig(Xs + t[1] * Us))) )
      dF = t -> [ sum( sum(c .* db)
                       for (c, db) in zip(C, ACE.evaluate_d(basis, ACEConfig(Xs + t[1] * Us)) * Us) ) ]
      print_tf(@test fdtest(F, dF, [0.0], verbose=false))
   end
   println()
end


#---
@info("SymmetricBasis construction and evaluation: Spherical Matrix")


for L1 = 0:1, L2 = 0:1
   @info "Tests for L₁ = $L1, L₂ = $L2 ⇿ $(get_orbsym(L1))-$(get_orbsym(L2)) block"
   φ = ACE.SphericalMatrix(L1, L2; T = ComplexF64)
   pibasis = PIBasis(B1p, ord, maxdeg; property = φ, isreal = false)
   basis = SymmetricBasis(pibasis, φ)
   BB = evaluate(basis, cfg)

   for ntest = 1:30
      Q, D1, D2 = ACE.Wigner.rand_QD(L1, L2)
      cfg1 = ACEConfig( shuffle(Ref(Q) .* Xs) )
      BB1 = evaluate(basis, cfg1)
      D1txBB1xD2 = Ref(D1') .* BB1 .* Ref(D2)
      print_tf(@test isapprox(D1txBB1xD2, BB, rtol=1e-10))
   end
   println()

   @info(" .... derivatives")
   for ntest = 1:30
      Us = randn(SVector{3, Float64}, length(Xs))
      C = randn(typeof(φ.val), length(basis))
      F = t -> sum( sum(c .* b.val)
                    for (c, b) in zip(C, ACE.evaluate(basis, ACEConfig(Xs + t[1] * Us))) )
      dF = t -> [ sum( sum(c .* db)
                       for (c, db) in zip(C, ACE.evaluate_d(basis, ACEConfig(Xs + t[1] * Us)) * Us) ) ]
      print_tf(@test fdtest(F, dF, [0.0], verbose=false))
   end
   println()
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

   for ntest = 1:30
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

for ntest = 1:30
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

end
