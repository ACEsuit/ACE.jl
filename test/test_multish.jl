


##

using ACE
using StaticArrays, Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis, 
           State 
using ACE.Random: rand_rot, rand_refl
using ACEbase.Testing: fdtest
using ACE.Testing: __TestSVec

rands3nrm() = ( rr = randn(SVector{3, Float64}); rr / norm(rr) )
rands3(rl, ru) = (rl + rand() * (ru-rl)) * rands3nrm()



## FIRST TEST (l, m) vs custom (lr, ms) 

# construct the 1p-basis
D = NaiveTotalDegree()
maxdeg = 6
ord = 3

B1p_r = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D, 
                                  varsym = :rr, idxsyms = (:nr, :lr, :mr))
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D, )
@show ACE.symbols(B1p_r)
@show ACE.indexrange(B1p_r)
println(@test all( ACE.indexrange(B1p_r)[symr] == ACE.indexrange(B1p)[sym]
                   for (symr, sym) in ((:nr, :n), (:lr, :l), (:mr, :m)) ) )
φ = ACE.Invariant()
basis = SymmetricBasis(φ, B1p, ACE.O3(), ord, maxdeg; Deg = D)
basis_r = SymmetricBasis(φ, B1p_r, ACE.O3(:lr, :mr), ord, maxdeg; Deg = D)

for ntest = 1:30 
   ru = basis.pibasis.basis1p.bases[1].R.ru 
   rl = basis.pibasis.basis1p.bases[1].R.rl 
   cfg = ACEConfig( [ State(rr = rands3(rl, ru)) for _=1:10 ] )
   print_tf(@test evaluate(basis, cfg) ≈ evaluate(basis_r, cfg))
end 

##

X = State( rr = rands3(), ss = rands3nrm() )
MagState = typeof(X)
Base.rand(::Type{MagState}) = MagState( rr = rands3(), ss = rands3nrm() )
Base.rand(::Type{MagState}, N::Integer) = [ rand(MagState) for _ = 1:N ]

B1p_r = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D, 
                                  varsym = :rr, idxsyms = (:nr, :lr, :mr))
B1p_s = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D, 
                                  varsym = :ss, idxsyms = (:ns, :ls, :ms))
B1p = B1p_r * B1p_s
@show ACE.symbols(B1p)
@show ACE.indexrange(B1p)


ACE.init1pspec!(B1p; maxdeg=maxdeg, Deg = D)
length(B1p)
ACE.get_spec(B1p)

# generate a configuration
nX = 10
Xs = rand(MagState, nX)
cfg = ACEConfig(Xs)
evaluate(B1p, Xs[1])
evaluate(B1p, cfg)

##

@info("SymmetricBasis construction and evaluation: Invariant Scalar")

φ = ACE.Invariant()
pibasis = PIBasis(B1p, ord, maxdeg; property = φ)

basis = SymmetricBasis(pibasis, φ)
@time SymmetricBasis(pibasis, φ);

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
      local BB1 = evaluate(basis, ACEConfig(Xs1))
      print_tf(@test isapprox(BB, BB1, rtol=1e-10))
end
println()

##
import ACEbase
@info("Test FIO")
let basis1 = basis 
   println(@test(all(ACEbase.Testing.test_fio(basis1; warntype=true))))
end


## 

@info("Test linear independence of the basis")
# generate some random configurations; ord^2 + 1 sounds good :)
cfgs = [ ACEConfig(rand(PositionState{Float64}, B1p.bases[1], nX)) 
         for _ = 1:(3*length(basis)) ]
A = zeros(length(cfgs), length(basis))
for (i, cfg) in enumerate(cfgs)
   A[i, :] = getproperty.(evaluate(basis, cfg), :val)
end
println(@test rank(A) == length(basis))

# ## Keep for futher profiling
# φ = ACE.Invariant()
# pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
# basis = SymmetricBasis(pibasis, φ)
# @time SymmetricBasis(pibasis, φ);
#
# Profile.clear()
# @profile SymmetricBasis(pibasis, φ);
# ProfileView.view()
#

## Testing derivatives


for ntest = 1:30
   _rrval(x::ACE.XState) = x.rr
   Us = randn(SVector{3, Float64}, length(Xs))
   c = randn(length(basis))
   F = t -> sum(c .* ACE.evaluate(basis, ACEConfig(Xs + t[1] * Us))).val
   dF = t -> [ Us' * _rrval.(sum(c .* ACE.evaluate_d(basis, ACEConfig(Xs + t[1] * Us)), dims=1)[:]) ]
   print_tf(@test fdtest(F, dF, [0.0], verbose=false))
end
println()

##
@info("SymmetricBasis construction and evaluation: Spherical Vector")

for L = 0:3
   @info "Tests for L = $L ⇿ $(get_orbsym(0))-$(get_orbsym(L)) block"
   local φ, pibasis, basis, BB, Iz
   φ = ACE.SphericalVector(L; T = ComplexF64)
   pibasis = PIBasis(B1p, ord, maxdeg; property = φ, isreal = false)
   basis = SymmetricBasis(pibasis, φ)
   @time SymmetricBasis(pibasis, φ);
   BB = evaluate(basis, cfg)

   Iz = findall(iszero, sum(norm, basis.A2Bmap, dims = 1))
   if !isempty(Iz)
      @warn("The A2B map for SphericalVector has $(length(Iz))/$(length(basis.pibasis)) zero-columns!!!!")
   end

   for ntest = 1:30
      local Q, D, BB1 
      Q, D = ACE.Wigner.rand_QD(L)
      cfg1 = ACEConfig( shuffle(Ref(Q) .* Xs) )
      BB1 = evaluate(basis, cfg1)
      DtxBB1 = Ref(D') .* BB1
      print_tf(@test isapprox(DtxBB1, BB, rtol=1e-10))
   end
   println()

   @info(" .... derivatives")
   for ntest = 1:30
      _rrval(x::ACE.XState) = x.rr
      Us = __TestSVec.(randn(SVector{3, Float64}, length(Xs)))
      C = randn(typeof(φ.val), length(basis))
      F = t -> sum( sum(c .* b.val)
                    for (c, b) in zip(C, ACE.evaluate(basis, ACEConfig(Xs + t[1] * Us))) )
      dF = t -> [ sum( sum(c .* db)
                  for (c, db) in zip(C, _rrval.(ACE.evaluate_d(basis, ACEConfig(Xs + t[1] * Us))) * Us) ) ]
      print_tf(@test fdtest(F, dF, [0.0], verbose=false))
   end
   println()
end

# ## Keep for futher profiling
# L = 1
# φ = ACE.SphericalVector(L; T = ComplexF64)
# pibasis = PIBasis(B1p, 4, 8; property = φ, isreal = false)
# basis = SymmetricBasis(pibasis, φ)
# @time SymmetricBasis(pibasis, φ);
#
# Profile.clear(); # Profile.init(; delay = 0.0001)
# @profile SymmetricBasis(pibasis, φ);
# ProfileView.view()

##

@info("SymmetricBasis construction and evaluation: Spherical Matrix")


for L1 = 0:2, L2 = 0:2
   @info "Tests for L₁ = $L1, L₂ = $L2 ⇿ $(get_orbsym(L1))-$(get_orbsym(L2)) block"
   local φ, pibasis, basis, BB, Iz
   φ = ACE.SphericalMatrix(L1, L2; T = ComplexF64)
   pibasis = PIBasis(B1p, ord, maxdeg; property = φ, isreal = false)
   basis = SymmetricBasis(pibasis, φ)
   @time basis = SymmetricBasis(pibasis, φ)
   BB = evaluate(basis, cfg)

   for ntest = 1:30
      local Q, D1, D2, BB1 
      Q, D1, D2 = ACE.Wigner.rand_QD(L1, L2)
      cfg1 = ACEConfig( shuffle(Ref(Q) .* Xs) )
      BB1 = evaluate(basis, cfg1)
      D1txBB1xD2 = Ref(D1') .* BB1 .* Ref(D2)
      print_tf(@test isapprox(D1txBB1xD2, BB, rtol=1e-10))
   end
   println()

   @info(" .... derivatives")
   for ntest = 1:30
      _rrval(x::ACE.XState) = x.rr
      Us = __TestSVec.(randn(SVector{3, Float64}, length(Xs)))
      C = randn(typeof(φ.val), length(basis))
      F = t -> sum( sum(c .* b.val)
                    for (c, b) in zip(C, ACE.evaluate(basis, ACEConfig(Xs + t[1] * Us))) )
      dF = t -> [ sum( sum(c .* db)
                       for (c, db) in zip(C, _rrval.(ACE.evaluate_d(basis, ACEConfig(Xs + t[1] * Us))) * Us) ) ]
      print_tf(@test fdtest(F, dF, [0.0], verbose=false))
   end
   println()
end


##
@info("Consistency between SphericalVector & SphericalMatrix")

for L = 0:3
   @info "L = $L"
   local Xs, cfg 
   φ1 = ACE.SphericalVector(L; T = ComplexF64)
   pibasis1 = PIBasis(B1p, ord, maxdeg; property = φ1, isreal = false)
   basis1 = SymmetricBasis(pibasis1, φ1)
   φ2 = ACE.SphericalMatrix(L, 0; T = ComplexF64)
   pibasis2 = PIBasis(B1p, ord, maxdeg; property = φ2, isreal = false)
   basis2 = SymmetricBasis(pibasis2, φ2)

   for ntest = 1:30
      Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
      cfg = ACEConfig(Xs)
      BBvec = evaluate(basis1, cfg)
      value1 = [reshape(BBvec[i].val, 2L+1, 1) for i in 1:length(BBvec)]
      BBmat = evaluate(basis2, cfg)
      value2 = [BBmat[i].val for i in 1:length(BBvec)]
      print_tf(@test isapprox(value1, value2, rtol=1e-10))
   end
   println()
end

##
@info("Consistency between Invariant Scalar & SphericalMatrix")

φ = ACE.Invariant()
pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
basis = SymmetricBasis(pibasis, φ)
φ2 = ACE.SphericalMatrix(0, 0; T = ComplexF64)
pibasis2 = PIBasis(B1p, ord, maxdeg; property = φ2, isreal = false)
basis2 = SymmetricBasis(pibasis2, φ2)

for ntest = 1:30
   local Xs, cfg, BB 
   Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
   cfg = ACEConfig(Xs)
   BB = evaluate(basis, cfg)
   BBsca = [BB[i].val for i in 1:length(BB)]
   BB2 =  evaluate(basis2, cfg)
   BBCFlo = [ComplexF64(BB2[i].val...) for i in 1:length(BB2)]
   print_tf(@test isapprox(BBsca, BBCFlo, rtol=1e-10))
end
println()

# ## Keep for futher profiling
#
# L1 = 1; L2 = 1
# φ = ACE.SphericalMatrix(L1, L2; T = ComplexF64)
# pibasis = PIBasis(B1p, ord, maxdeg; property = φ, isreal = false)
# basis = SymmetricBasis(pibasis, φ)
# @time SymmetricBasis(pibasis, φ);
#
# Profile.clear()
# @profile SymmetricBasis(pibasis, φ);
# ProfileView.view()


##
