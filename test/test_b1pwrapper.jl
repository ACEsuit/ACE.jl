##

using ACE
using Printf, Test, LinearAlgebra, StaticArrays
using ACE: evaluate, evaluate_d, Rn1pBasis, Ylm1pBasis,
      PositionState, Product1pBasis, getlabel
using Random: shuffle
using ACEbase.Testing: dirfdtest, fdtest, print_tf, test_fio, println_slim

##

@info "Build a 1p basis from scratch"

maxdeg = 5
r0 = 1.0
rcut = 3.0
maxorder = 3
Bsel = SimpleSparseBasis(maxorder, maxdeg)

## check radial basis 

trans = PolyTransform(1, r0)   # r -> x = 1/r^2
J = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)   #  J_n(x) * (x - xcut)^pcut

# old implementation 
Rn = Rn1pBasis(J; label = "Rn")

# new implementation 
Rn_w = ACE.Rn1pBasis_w(J)

for ntest = 1:30 
   X = ACE.State(rr = ACE.rand_radial(J) * ACE.rand_sphere())
   print_tf(@test evaluate(Rn, X) ≈ evaluate(Rn_w, X) )
end

##

Ylm = Ylm1pBasis(maxdeg; label = "Ylm")

Ylm_w = ACE.Ylm1pBasis_w(Ylm.SH.alp.L)

for ntest = 1:30 
   X = ACE.State(rr = ACE.rand_radial(J) * ACE.rand_sphere())
   print_tf(@test evaluate(Ylm, X) ≈ evaluate(Ylm_w, X) )
end


## scalar basis 

Bu = ACE.Scal1pBasis(:u, nothing, :k, J) 
Bu_w = ACE.Scal1pBasis_w(:u, nothing, :k, J)

for ntest = 1:30 
   X = ACE.State(rr = ACE.rand_radial(J) * ACE.rand_sphere(), u = ACE.rand_radial(J))
   print_tf(@test evaluate(Bu, X) ≈ evaluate(Bu_w, X) )
end


