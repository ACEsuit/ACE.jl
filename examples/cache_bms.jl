
##
using ACE, Printf, Test, LinearAlgebra
using ACE: evaluate, evaluate_d, read_dict, write_dict
using ACE.Testing

verbose = false
maxdeg = 25

using BenchmarkTools
pin = 0; pcut = 2
trans = PolyTransform(2, 1.0)
B = transformed_jacobi(maxdeg, trans, 3.0, 0.5, pin = pin, pcut = pcut)
Rn_new = ACE.Rn1pBasis_new(B.J; trans=trans)
X = State( rr = ACE.rand_sphere() * ACE.rand_radial(B) )
evaluate(Rn_new, X)

function runn(f, Rn, X, N)
   for n = 1:N 
      B = f(Rn, X)
      ACE.release!(B)
   end
   return nothing 
end

@btime runn($evaluate, $Rn_new, $X, 1_000)
@btime runn($evaluate, $(Rn_new.basis), $(X.rr), 1_000)


function runn!(B, Rn, X, N, trans)
   for n = 1:N 
      B = ACE.evaluate!(B, Rn, trans(X))
      ACE.release!(B)
   end
   return nothing 
end

tmpc = evaluate(Rn_new, X)
tmp = copy(tmpc.A)
@btime runn!($tmp, $(B.J), $(norm(X.rr)), 1_000, $trans)
@btime runn!($tmpc, $(B.J), $(norm(X.rr)), 1_000, $trans)


##
@profview runn(evaluate, Rn_new.basis, X.rr, 100_000_000)

@profview runn!(tmp, B.J, norm(X.rr), 100_000_000, trans)

## 
using Profile 
Profile.clear() 
@profile runn(evaluate, Rn_new.basis, X.rr, 10_000_000)
Profile.print()

##
Rn_old = ACE.chain(trans, B.J)


let B0 = Rn_old, B1 = Rn_new.basis, X=X, B2 = B.J, tmpc=tmpc, tmp=tmp, trans=trans
   N = 10_000_000
   @time runn(evaluate, B0, norm(X.rr), N)
   @time runn(evaluate, B0, norm(X.rr), N)
   @time runn(evaluate, B1, X.rr, N)
   @time runn(evaluate, B1, X.rr, N)
   @time runn!(tmp, B2, norm(X.rr), N, trans)
   @time runn!(tmp, B2, norm(X.rr), N, trans)
end
