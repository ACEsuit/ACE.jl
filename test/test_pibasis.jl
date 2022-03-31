



##


using ACE, Random
using Printf, Test, LinearAlgebra, ACE.Testing, StaticArrays
using ACEbase.Testing: dirfdtest, fdtest, print_tf, test_fio, println_slim 
using ACE: evaluate, evaluate_d, Rn1pBasis, Ylm1pBasis,
      PositionState, Product1pBasis, O3, rand_vec3

##

@info("Basic test of PIBasis construction and evaluation")

maxdeg = 6
ord = 3
Bsel = SimpleSparseBasis(ord, maxdeg) 

φ = ACE.Invariant()

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)

pibasis = PIBasis(B1p, O3(), Bsel; property = φ)
pibasis_r = PIBasis(B1p, O3(), Bsel; property = φ, isreal=true)

# generate a configuration
nX = 10
_randX() = State(rr = rand_vec3(B1p["Rn"]) )
Xs = [_randX() for _=1:nX]
cfg = ACEConfig(Xs)

AA = evaluate(pibasis, cfg)
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

@warn("Turned off failing FIO tests")
@info("FIO Test")
# println_slim(@test( all(test_fio(pibasis)) ))
# println_slim(@test( all(test_fio(pibasis_r)) ))

## Testing derivatives

@info("Derivatives of PIbasis")
for (pibasis, AA) in [(pibasis, AA), (pibasis_r, AA_r)]
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


