


##


using ACE, StaticArrays
using Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis
using ACE.Random: rand_rot, rand_refl
using ACEbase.Testing: fdtest

# construct the 1p-basis
D = NaiveTotalDegree()
maxdeg = 6
ord = 3

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

# generate a configuration
nX = 10
Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
cfg = ACEConfig(Xs)

##

@info("SymmetricBasis construction and evaluation: EuclideanVector")


φ = ACE.EuclideanVector(Complex{Float64})
pibasis = PIBasis(B1p, O3(), ord, maxdeg; property = φ, isreal=false)
basis = SymmetricBasis(φ, O3(), pibasis; isreal=true)
@time SymmetricBasis(φ, O3(), pibasis; isreal=true)

BB = evaluate(basis, cfg)

Iz = findall(iszero, sum(norm, basis.A2Bmap, dims=1)[:])
if !isempty(Iz)
   @warn("The A2B map for EuclideanVector has $(length(Iz))/$(length(basis.pibasis)) zero-columns!!!!")
end

##

@info("Test FIO")
using ACEbase.Testing: test_fio

println(@test(all(test_fio(basis; warntype = false))))


##

@info("Test equivariance properties for real version")

tol = 1e-10

@info("check for rotation, permutation and inversion equivariance")
for ntest = 1:30
   local Xs, BB
   Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
   BB = evaluate(basis, ACEConfig(Xs))
   Q = rand([-1,1]) * ACE.Random.rand_rot()
   Xs_rot = Ref(Q) .* shuffle(Xs)
   BB_rot = evaluate(basis, ACEConfig(Xs_rot))
   print_tf(@test all([ norm(Q' * b1 - b2) < tol
                        for (b1, b2) in zip(BB_rot, BB)  ]))
end
println()


@info("Test equivariance properties for complex version")

basis = SymmetricBasis(φ, O3(), pibasis; isreal=false)
# a stupid but necessary test
BB = evaluate(basis, cfg)
BB1 = basis.A2Bmap * evaluate(basis.pibasis, cfg)
println(@test isapprox(BB, BB1, rtol=1e-10)) # MS: This test will fail for isreal=true


@info("check for rotation, permutation and inversion equivariance")
for ntest = 1:30
   local Xs, BB
   Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
   BB = evaluate(basis, ACEConfig(Xs))
   Q = rand([-1,1]) * ACE.Random.rand_rot()
   Xs_rot = Ref(Q) .* shuffle(Xs)
   BB_rot = evaluate(basis, ACEConfig(Xs_rot))
   print_tf(@test all([ norm(Q' * b1 - b2) < tol
                        for (b1, b2) in zip(BB_rot, BB)  ]))
end
println()

# ## keep for further profiling
#
# φ = ACE.EuclideanVector(Complex{Float64})
# pibasis = PIBasis(B1p, ord, maxdeg; property = φ, isreal = false)
# basis = SymmetricBasis(pibasis, φ)
# @time SymmetricBasis(pibasis, φ);
#
# Profile.clear(); # Profile.init(; delay = 0.0001)
# @profile SymmetricBasis(pibasis, φ);
# ProfileView.view()

##

@info(" ... derivatives")
_rrval(x::ACE.XState) = x.rr
for ntest = 1:30
   Us = randn(SVector{3, Float64}, length(Xs))
   C = randn(typeof(φ.val), length(basis))
   F = t -> sum( sum(c .* b.val)
                 for (c, b) in zip(C, ACE.evaluate(basis, ACEConfig(Xs + t[1] * Us))) )
   dF = t -> [ sum( sum(c .* db)
                    for (c, db) in zip(C, _rrval.(ACE.evaluate_d(basis, ACEConfig(Xs + t[1] * Us))) * Us) ) ]
   print_tf(@test fdtest(F, dF, [0.0], verbose=false))
end
println()

##

