##

using ACE
using Printf, Test, LinearAlgebra, StaticArrays
using ACE: evaluate, evaluate_d, Rn1pBasis, Ylm1pBasis,
      PositionState, Product1pBasis
using Random: shuffle
using ACEbase.Testing: dirfdtest, fdtest, print_tf, test_fio, 
                       println_slim

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


const mult_val = 1.234
mult1 = M.TestMult(X -> mult_val)

B1p = mult1 * RnYlm
ACE.init1pspec!(B1p, Bsel)

println_slim(@test length(B1p) == length(RnYlm))

nX = 10
Xs = rand(PositionState{Float64}, RnYlm.bases[1], nX)
cfg = ACEConfig(Xs)

A1 = evaluate(RnYlm, cfg)
A2 = evaluate(B1p, cfg)
println_slim(@test( A1 * mult_val ≈ A2 ))

##

@info("test against manual summation")

_f = X -> exp(- ACE.sumsq(X.rr .- mult_val))
mult2 = M.TestMult(_f)
B1p2 = mult2 * RnYlm
ACE.init1pspec!(B1p2, Bsel)

A1 = sum( evaluate(RnYlm, X) * _f(X) for X in Xs )
A2 = evaluate(B1p2, cfg)
println_slim(@test A1 ≈ A2)


##

using ACE: CylindricalBondEnvelope

r0cut = 8.0
rcut = 4.0
zcut = 2.0
env = CylindricalBondEnvelope(r0cut, rcut, zcut)

B1p = mult1 * RnYlm
ACE.init1pspec!(B1p, Bsel)

Xs = rand(PositionState{Float64}, RnYlm.bases[1], nX)

evaluate(B1p, Xs[1])

##

using ACE: Invariant, SymmetricBasis
Bsym = SymmetricBasis(Invariant(), B1p, Bsel)

evaluate(Bsym, ACEConfig(Xs))

