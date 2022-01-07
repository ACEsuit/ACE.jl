


##

using ACE
using StaticArrays, Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, SymmetricBasis, PIBasis, O3
using ACE.Random: rand_rot, rand_refl
using ACEbase.Testing: fdtest, println_slim
using ACE.Testing: __TestSVec

# using Profile, ProfileView

# Extra using Wigner for computing Wigner Matrix
using ACE.Wigner
using ACE.Wigner: get_orbsym


# construct the 1p-basis
maxdeg = 6
ord = 3
Bsel = SimpleSparseBasis(ord, maxdeg)

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)

# generate a configuration
nX = 10
Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
cfg = ACEConfig(Xs)

##

@info("SymmetricBasis construction and evaluation: Invariant Scalar")

φ = ACE.Invariant()
basis = SymmetricBasis(φ, B1p, Bsel)
@time SymmetricBasis(φ, B1p, Bsel);

BB = evaluate(basis, cfg)

# a stupid but necessary test
AA = evaluate(basis.pibasis, cfg)
BB1 = basis.A2Bmap * AA
println_slim(@test isapprox(BB, BB1, rtol=1e-10))

@info("evaluate with vector vs config")
println_slim(@test BB ≈ evaluate(basis, Xs))
println_slim(@test evaluate_d(basis, cfg) ≈ evaluate_d(basis, Xs))


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

@info("Test Invariant Basis without Constant")
basis_noc = SymmetricBasis(φ, B1p, Bsel; filterfun = ACE.NoConstant())
println_slim(@test length(basis_noc) == length(basis) - 1)
BB_noc = evaluate(basis_noc, cfg)
println_slim(@test (BB_noc == BB[2:end]))

##

@info("Test what happens with an empty configuration")

Xs_empty = Vector{eltype(Xs)}(undef, 0)
cfg_empty = ACEConfig(Xs_empty)
B_empty = evaluate(basis, cfg_empty)
println(@test( all(iszero, B_empty[2:end]) ))
println(@test( B_empty[1] == ACE.Invariant(1.0) ) )

##
import ACEbase
@info("Test FIO")
let basis1 = basis 
   println_slim(@test(all(ACEbase.Testing.test_fio(basis1; warntype=true))))
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
println_slim(@test rank(A) == length(basis))

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
   basis = SymmetricBasis(φ, B1p, Bsel)
   @time SymmetricBasis(φ, B1p, Bsel)
   BB = evaluate(basis, cfg)

   @info("check that no-const gives the same (sub-)basis")
   basis0 = SymmetricBasis(φ, B1p, Bsel; filterfun = ACE.NoConstant())
   BB0 = evaluate(basis0, cfg)
   if L == 0 
      println_slim(@test BB[2:end] ≈ BB0)   
   else
      println_slim(@test BB ≈ BB0)
   end

   Iz = findall(iszero, sum(norm, basis.A2Bmap, dims = 1))
   if !isempty(Iz)
      @warn("The A2B map for SphericalVector has $(length(Iz))/$(length(basis.pibasis)) zero-columns!!!!")
   end

   @info("check symmetry ")
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
   basis = SymmetricBasis(φ, B1p, Bsel)
   @time basis = SymmetricBasis(φ, B1p, Bsel)
   BB = evaluate(basis, cfg)

   @info("check that no-const gives the same (sub-)basis")
   basis0 = SymmetricBasis(φ, B1p, Bsel; filterfun = ACE.NoConstant())
   BB0 = evaluate(basis0, cfg)
   if L1 == L2  
      println_slim(@test BB[2:end] ≈ BB0)
   else
      println_slim(@test BB ≈ BB0)
   end

   @info("check symmetry ")
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
   basis1 = SymmetricBasis(φ1, B1p, Bsel)

   φ2 = ACE.SphericalMatrix(L, 0; T = ComplexF64)
   basis2 = SymmetricBasis(φ2, B1p, Bsel)

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
basis = SymmetricBasis(φ, B1p, Bsel)
φ2 = ACE.SphericalMatrix(0, 0; T = ComplexF64)
basis2 = SymmetricBasis(φ2, B1p, Bsel)

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
