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
Rn_w = ACE.B1pComponent(J, 
                       ACE.GetNorm{:rr}(), 
                       [ (n = i,) for i = 1:length(J) ], 
                       "Rn_w"
                       )

for ntest = 1:30 
   X = ACE.State(rr = ACE.rand_radial(J) * ACE.rand_sphere())
   print_tf(@test evaluate(Rn, X) ≈ evaluate(Rn_w, X) )
end

##

Ylm = Ylm1pBasis(maxdeg; label = "Ylm")


get_ylm_spec(SH) = [ begin l, m = ACE.SphericalHarmonics.idx2lm(i); 
                      (l = l, m = m) end for i = 1:length(SH) ]

ylm_spec = get_ylm_spec(Ylm.SH)

Ylm_w = ACE.B1pComponent(Ylm.SH, ACE.GetVal{:rr}(), ylm_spec, "Ylm_w")

for ntest = 1:30 
   X = ACE.State(rr = ACE.rand_radial(J) * ACE.rand_sphere())
   print_tf(@test evaluate(Ylm, X) ≈ evaluate(Ylm_w, X) )
end


## scalar basis 

Bu = ACE.Scal1pBasis(:u, nothing, :k, J) 
Bu_w = ACE.B1pComponent(J, ACE.GetVal{:u}(), [ (k=i,) for i = 1:length(J)], "Bu_w")

for ntest = 1:30 
   X = ACE.State(rr = ACE.rand_radial(J) * ACE.rand_sphere(), u = ACE.rand_radial(J))
   print_tf(@test evaluate(Bu, X) ≈ evaluate(Bu_w, X) )
end


