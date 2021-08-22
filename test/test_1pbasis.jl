

##

using ACE
using Printf, Test, LinearAlgebra, StaticArrays
using ACE: evaluate, evaluate_d, Rn1pBasis, Ylm1pBasis,
      PositionState, Product1pBasis
using Random: shuffle
using ACEbase.Testing: dirfdtest, fdtest, print_tf, test_fio

##


@info "Build a 1p basis from scratch"

maxdeg = 5
r0 = 1.0
rcut = 3.0
maxorder = 3
Bsel = SimpleSparseBasis(maxorder, maxdeg)

trans = PolyTransform(1, r0)   # r -> x = 1/r^2
J = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)   #  J_n(x) * (x - xcut)^pcut
Rn = Rn1pBasis(J)
Ylm = Ylm1pBasis(maxdeg)
B1p = Product1pBasis( (Rn, Ylm) )
ACE.init1pspec!(B1p, Bsel)

nX = 10
Xs = rand(PositionState{Float64}, Rn, nX)
cfg = ACEConfig(Xs)

A = evaluate(B1p, cfg)

@info("test against manual summation")
A1 = sum( evaluate(B1p, X) for X in Xs )
println(@test A1 ≈ A)

@info("test permutation invariance")
for ntest = 1:30
   print_tf(@test A ≈ evaluate(B1p, ACEConfig(shuffle(Xs))))
end
println()

##

# test_fio(Ylm)
# D = ACE.write_dict(Ylm)
# Ylm_ = ACE.read_dict(D)
# Ylm.SH.alp == Ylm_.SH.alp
# Ylm.SH.alp.B ≈ Ylm_.SH.alp.B
# @which ( Ylm.SH.alp == Ylm_.SH.alp )

@info("Test FIO")
for _B in (J, Rn, Ylm, B1p)
   println((@test(all(test_fio(_B)))))
end

##

@info("Ylm1pBasis gradients")
Y = ACE.acquire_B!(Ylm, Xs[1])
dY = ACE.acquire_dB!(Ylm, Xs[1])
println(@test (typeof(dY) == eltype(Ylm.dB_pool.arrays)))
ACE.evaluate!(Y, Ylm, Xs[1])
ACE.evaluate_d!(dY, Ylm, Xs[1])
Y1 = ACE.acquire_B!(Ylm, Xs[1])
dY1 = ACE.acquire_dB!(Ylm, Xs[1])
ACE.evaluate_ed!(Y1, dY1, Ylm, Xs[1])

println(@test (evaluate(Ylm, Xs[1]) ≈ Y))
println(@test (evaluate_d(Ylm, Xs[1]) ≈ dY))
println(@test ((Y ≈ Y1) && (dY ≈ dY1)) )

_vec2X(x) = PositionState{Float64}((rr = SVector{3}(x),))

for ntest = 1:30
   x0 = randn(3)
   c = rand(length(Y))
   F = x -> sum(ACE.evaluate(Ylm, _vec2X(x)) .* c)
   dF = x -> sum(ACE.evaluate_d(Ylm, _vec2X(x)) .* c).rr |> Vector
   print_tf(@test fdtest(F, dF, x0; verbose=false))
end
println()
##

@info("Rn1pBasis gradients")

for ntest = 1:30
   x0 = randn(3)
   c = rand(length(Rn))
   F = x -> sum(ACE.evaluate(Rn, _vec2X(x)) .* c)
   dF = x -> sum(ACE.evaluate_d(Rn, _vec2X(x)) .* c).rr |> Vector
   print_tf(@test fdtest(F, dF, x0; verbose=false))
end
println()

##

@info("Product basis evaluate_ed! tests")

A1 = ACE.acquire_B!(B1p, cfg)
ACE.evaluate!(A1, B1p, cfg)
A2 = ACE.acquire_B!(B1p, cfg)
dA = ACE.acquire_dB!(B1p, cfg)
ACE.evaluate_ed!(A2, dA, B1p, cfg)
println(@test A1 ≈ A2)

println(@test( evaluate_d(B1p, cfg) ≈ dA ))

##
@info("Product basis gradient test")

for ntest = 1:30
   x0 = randn(3)
   c = rand(length(B1p))
   F = x -> sum(ACE.evaluate(B1p, _vec2X(x)) .* c)
   dF = x -> sum(ACE.evaluate_d(B1p, ACEConfig([_vec2X(x)])) .* c).rr |> Vector
   print_tf(@test fdtest(F, dF, x0; verbose=false))
end
println()
##

