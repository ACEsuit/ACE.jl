using ACE, StaticArrays
using Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, SymmetricBasis, PIBasis, EuclideanMatrix
using ACE.Random: rand_rot, rand_refl
using ACEbase.Testing: fdtest

## construct the 1p-basis
maxdeg = 6
ord = 3
Bsel = SimpleSparseBasis(ord, maxdeg)

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)

# generate a configuration
nX = 10
Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
cfg = ACEConfig(Xs)

##

@info("SymmetricBasis construction and evaluation: EuclideanMatrix")


φ = ACE.EuclideanMatrix(Float64)
pibasis = PIBasis(B1p, Bsel; property = φ)
basis = SymmetricBasis(φ, pibasis)
@time SymmetricBasis(φ, pibasis)

BB = evaluate(basis, cfg)

Iz = findall(iszero, sum(norm, basis.A2Bmap, dims=1)[:])
if !isempty(Iz)
   @warn("The A2B map for EuclideanMatrix has $(length(Iz))/$(length(basis.pibasis)) zero-columns!!!!")
end

##

@info("Test FIO")
using ACEbase.Testing: test_fio

println(@test(all(test_fio(basis; warntype = false))))

##

@info("Test equivariance properties for real version")

tol = 1e-12

##
#                     for (b1, b2) in zip(BB_rot, BB)  
#print_tf(@test all([ norm(Q' * b1 * Q - b2) < tol
#                     for (b1, b2) in zip(BB_rot, BB)  ]))
##                      
@info("check for rotation, permutation and inversion equivariance")
for ntest = 1:30
   local Xs, BB
   Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
   BB = evaluate(basis, ACEConfig(Xs))
   Q = rand([-1,1]) * ACE.Random.rand_rot()
   Xs_rot = Ref(Q) .* shuffle(Xs)
   BB_rot = evaluate(basis, ACEConfig(Xs_rot))
   print_tf(@test all([ norm(Q' * b1 * Q - b2) < tol
                        for (b1, b2) in zip(BB_rot, BB)  ]))
end
println()

@info("Check for some non-symmetric matrix functions")
for ntest = 1:30
   local Xs, BB
   Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
   BB = evaluate(basis, ACEConfig(Xs))
   print_tf(@test any([ b.val != transpose(b.val)
                        for b in BB  ]))
end
println()

##

imtol = 5.0
@info("Check magnitude of complex part")
for ntest = 1:30
   local Xs, BB
   Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
   BB = evaluate(basis, ACEConfig(Xs))
   for (i,b) in enumerate(BB)
      if norm(imag(b.val)) > imtol
         @warn( "Large imaginary part for $(ACE.get_spec(basis)[i]), $(norm(imag(b.val)))")
      end
   end
   #println(maximum([ norm(imag(b.val))/ norm(real(b.val))  for b in BBs  ]))
   #print_tf(@test all([ norm(imag(b.val)) < .1  for b in BB  ]))
end
println()
print(ACE.get_spec(basis)[1])
##

@info("Test equivariance properties for complex version")

basis = SymmetricBasis(φ, pibasis; isreal=false)
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
   print_tf(@test all([ norm(Q' * b1 * Q - b2) < tol
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

#=
@info(" ... derivatives")
_rrval(x::ACE.XState) = x.rr
for ntest = 1:30
   Us = randn(SVector{3,Float64 }, length(Xs))
   C = randn(typeof(φ.val), length(basis))
   F = t -> sum( sum(c .* b.val)
                 for (c, b) in zip(C, ACE.evaluate(basis, ACEConfig(Xs + t[1] * Us))) )
   dF = t -> [ sum( sum(c .* db)
                    for (c, db) in zip(C, _rrval.(ACE.evaluate_d(basis, ACEConfig(Xs + t[1] * Us))) * Us) ) ]
   print_tf(@test fdtest(F, dF, [0.0], verbose=false))
end
println()
=#

##
