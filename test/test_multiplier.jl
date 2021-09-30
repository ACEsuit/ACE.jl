##

using ACE
using Printf, Test, LinearAlgebra, StaticArrays
using ACE: evaluate, evaluate_d, Rn1pBasis, Ylm1pBasis,
      PositionState, Product1pBasis
using Random: shuffle
using ACEbase.Testing: dirfdtest, fdtest, print_tf, test_fio

##
@info("Testing B1pMultiplier")

module M
    import ACE 
    struct TestMult{TF} <: ACE.B1pMultiplier{Float64}
        f::TF
    end

    ACE._inner_evaluate(mult::TestMult, X) = mult.f(X)
end

Bsel = SimpleSparseBasis(3, 5)

RnYlm = ACE.Utils.RnYlm_1pbasis()
ACE.init1pspec!(RnYlm, Bsel)

@info("some basic tests")


const val1 = 1.234
mult1 = M.TestMult(X -> val1)

B1p = mult1 * RnYlm
ACE.init1pspec!(B1p, Bsel)

println(@test length(B1p) == length(RnYlm))

nX = 10
Xs = rand(PositionState{Float64}, RnYlm.bases[1], nX)
cfg = ACEConfig(Xs)

A1 = evaluate(RnYlm, cfg)
A2 = evaluate(B1p, cfg)
println(@test( A1 * val1 ≈ A2 ))

##

@info("test against manual summation")

_f = X -> exp(-sum(abs2, X.rr .- val1))
mult2 = M.TestMult(_f)
B1p2 = mult2 * RnYlm
ACE.init1pspec!(B1p2, Bsel)

A1 = sum( evaluate(RnYlm, X) * _f(X) for X in Xs )
A2 = evaluate(B1p2, cfg)
println(@test A1 ≈ A2)


##
