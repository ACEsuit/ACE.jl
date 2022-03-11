

##

using ACE
using Printf, Test, LinearAlgebra, StaticArrays
using ACE: evaluate, evaluate_d, Rn1pBasis, Ylm1pBasis,
      PositionState, Product1pBasis, State, ACEConfig, 
      SymmetricBasis, get_spec 
using Random: shuffle
using ACEbase.Testing: dirfdtest, fdtest, print_tf, test_fio
using ACE.OrthPolys: transformed_jacobi

##

maxdeg = 10
trans = ACE.Transforms.IdTransform()
P = transformed_jacobi(maxdeg, trans, 1.0, 0.0; pin = 0, pcut = 0) 
Bsel = ACE.SimpleSparseBasis(3, maxdeg)
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
symB = SymmetricBasis(ACE.Invariant(), B1p, Bsel)


##
@info("Attempt to sparsify a basis")
symB2 = deepcopy(symB)
spec = get_spec(symB2)
keep = findall(bb -> all(b.n <= 4 && b.l <= 3 for b in bb), spec)
ACE.sparsify!(symB2, keep = keep)

@show length(symB), length(symB2)
@show length(symB.pibasis), length(symB2.pibasis)
@show length(symB.pibasis.basis1p), length(symB2.pibasis.basis1p)

@info("some basic consistency checks")
print_tf(@test length(symB2) == length(keep))
print_tf(@test length(symB.pibasis) > length(symB2.pibasis))
print_tf(@test length(symB.pibasis.basis1p) > length(symB2.pibasis.basis1p))
println() 

## 

@info("check evaluation is consistent")

for ntest = 1:30 
   local cfg , BB1 
   cfg = [ State(rr = ACE.rand_radial(B1p["Rn"].basis) * ACE.rand_sphere()) for _=1:10 ] 
   BB1 = evaluate(symB, cfg)
   BB2 = evaluate(symB2, cfg)
   print_tf(@test BB1[keep] ≈ BB2)
end
println() 
