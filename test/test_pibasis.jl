



##


using ACE, Random
using Printf, Test, LinearAlgebra, ACE.Testing, StaticArrays
using ACEbase.Testing: dirfdtest, fdtest, print_tf, test_fio
using ACE: evaluate, evaluate_d, Rn1pBasis, Ylm1pBasis,
      PositionState, Product1pBasis, NaiveTotalDegree, O3

##

@info("Basic test of PIBasis construction and evaluation")

D = NaiveTotalDegree()
maxdeg = 6
ord = 3
φ = ACE.Invariant()

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

pibasis = PIBasis(B1p, O3(), ord, maxdeg; property = φ)

# generate a configuration
nX = 10
Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
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

## a really naive implementation of PIBasis to check correctness
A = evaluate(B1p, cfg)
AA_naive =  [
      real(prod( A[ inv_spec1[ b1 ] ] for b1 in b )) for b in spec ]
println(@test( AA_naive ≈ AA ))

## FIO tests 

@info("FIO Test")
println(@test( all(test_fio(pibasis)) ))

## Testing derivatives

@info("Derivatives of PIbasis")
AA1, dAA = ACE.evaluate_ed(pibasis, cfg)
println(@test AA1 ≈ AA)

##

for ntest = 1:30
  _rrval(x::ACE.XState) = x.rr
  Us = randn(SVector{3, Float64}, length(Xs))
  c = randn(length(pibasis))
  F = t -> sum(c .* ACE.evaluate(pibasis, ACEConfig(Xs + t[1] * Us)))
  dF = t -> [ Us' * _rrval.(sum(c .* ACE.evaluate_ed(pibasis, ACEConfig(Xs + t[1] * Us))[2], dims=1)[:]) ]
  print_tf(@test fdtest(F, dF, [0.0], verbose=false))
end
println()
##


