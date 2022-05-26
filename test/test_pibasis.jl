



##


using ACE, Random
using Printf, Test, LinearAlgebra, ACE.Testing, StaticArrays
using ACEbase.Testing: dirfdtest, fdtest, print_tf, test_fio, println_slim 
using ACE: evaluate, evaluate_d, Rn1pBasis, Ylm1pBasis,
      PositionState, Product1pBasis, O3, rand_vec3

##

@info("Basic test of PIBasis construction and evaluation")

maxdeg = 8
ord = 3
Bsel = SimpleSparseBasis(ord, maxdeg) 

φ = ACE.Invariant()

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)

pibasis = PIBasis(B1p, O3(), Bsel; property = φ)
pibasis_r = PIBasis(B1p, O3(), Bsel; property = φ, isreal=true)

# generate a configuration
nX = 40
_randX() = State(rr = rand_vec3(B1p["Rn"]) )
Xs = [_randX() for _=1:nX]
cfg = ACEConfig(Xs)

AA = evaluate(pibasis, cfg)
evaluate(pibasis, cfg) == evaluate(pibasis, Xs)

AA_r = evaluate(pibasis_r, cfg)

println_slim(@test(length(pibasis) == length(AA)))

spec = ACE.get_spec(pibasis)
println_slim(@test all(length(b) > 0 for b in spec[2:end]))
println_slim(@test length(spec[1]) == 0)  # the constant term 

spec_naive = [
    [ ACE.get_spec(B1p, pibasis.spec.iAA2iA[iAA, t])
      for t = 1:pibasis.spec.orders[iAA] ]   for iAA = 1:length(pibasis)
    ]

println_slim(@test spec == spec_naive)


# get inverse Aspec
inv_spec1 = Dict{Any, Int}()
for (i, b1) in enumerate(ACE.get_spec(B1p))
  inv_spec1[b1] = i
end

## a really naive implementation of PIBasis to check correctness
A = evaluate(B1p, cfg)
AA_naive =  [ prod( A[ inv_spec1[ b1 ] ] for b1 in b; init=1.0 ) for b in spec ]
println_slim(@test( AA_naive ≈ AA ))

AA_r_naive = real.(AA_naive)
println_slim(@test( AA_r_naive ≈ AA_r ))



## FIO tests 
@info("FIO Test")
println_slim(@test( all(test_fio(pibasis; warntype=false)) ))
println_slim(@test( all(test_fio(pibasis_r; warntype=false)) ))

## Testing derivatives

AA, dAA = ACE.evaluate_ed(pibasis, cfg)

##

@info("Derivatives of PIbasis")
for (pibasis, AA) in [(pibasis, AA), (pibasis_r, AA_r)]
  local AA, dAA 
  AA1, dAA = ACE.evaluate_ed(pibasis, cfg)
  println_slim(@test AA1 ≈ AA)

  for ntest = 1:30
    _rrval(x::ACE.XState) = x.rr
    Us = randn(SVector{3, Float64}, length(Xs))
    c = randn(length(pibasis))
    F = t -> sum(c .* ACE.evaluate(pibasis, ACEConfig(Xs + t[1] * Us)))
    dF = t -> [ Us' * _rrval.(sum(c .* ACE.evaluate_ed(pibasis, ACEConfig(Xs + t[1] * Us))[2], dims=1)[:]) ]
    print_tf(@test fdtest(F, dF, [0.0], verbose=false))
  end
  println()
end

##

import ACE: evaluate_ed 
@info("Test the chained version of PIBasis")
A = evaluate(pibasis.basis1p, cfg)
println_slim(@test evaluate(pibasis, A) == evaluate(pibasis, cfg))

A, dA = ACE.evaluate_ed(pibasis.basis1p, cfg)
println_slim(@test evaluate_ed(pibasis, A, dA) == evaluate_ed(pibasis, cfg))

##

# using BenchmarkTools

# A, dA = ACE.evaluate_ed(pibasis.basis1p, cfg)
# AA, dAA =  evaluate_ed(pibasis, A, dA)

# @btime evaluate_ed($pibasis, $Xs)
# @btime ACE.evaluate_ed!($AA, $dAA, $pibasis, $Xs)

# @profview begin 
#   for _=1:100 
#     ACE.evaluate_ed!(AA, dAA, pibasis, Xs)
#   end 
# end