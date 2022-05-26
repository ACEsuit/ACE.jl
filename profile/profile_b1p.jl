


using ACE
using Printf, Test, LinearAlgebra, StaticArrays, BenchmarkTools
using ACE: evaluate, evaluate_d, evaluate_ed, 
      Rn1pBasis, Ylm1pBasis,
      PositionState, Product1pBasis, getlabel, get_spec, 
      State, DState, rand_vec3, rand_radial, rand_sphere, Scal1pBasis, 
      valtype, gradtype, acquire_B!, acquire_dB!, 
      discrete_jacobi, release! 
using Random: shuffle
using ACEbase.Testing: dirfdtest, fdtest, print_tf, test_fio, println_slim

##


@info "Build a 1p basis from scratch"

maxdeg = 20
maxL = 10
r0 = 1.0
rcut = 3.0
maxorder = 3
Bsel = SimpleSparseBasis(maxorder, maxdeg)

trans = PolyTransform(1, r0)   # r -> x = 1/r^2
J = discrete_jacobi(maxdeg; pcut=2, xcut = rcut, pin = 0, xin = 0.0, trans=trans)
Rn = Rn1pBasis(J, trans)
Ylm = Ylm1pBasis(maxL)
Pk = Scal1pBasis(:u, nothing, :k, J)
A_nlm = Product1pBasis( (Rn, Ylm) )
ACE.init1pspec!(A_nlm, Bsel)
A_nlmk = Product1pBasis( (Rn, Ylm, Pk) )
ACE.init1pspec!(A_nlmk, Bsel)

nX = 10
Xs = [ State(rr = rand_vec3(Rn), u = randn() ) for _=1:nX ]
X = Xs[1]
cfg = ACEConfig(Xs)

# A = evaluate(A_nlm, Xs)

##

@info("Evaluate Pk, Ylm, Rn WITHOUT release")
@btime B = evaluate($Pk, $X)
@btime B = evaluate($Ylm, $X)
@btime B = evaluate($Rn, $X)

@info("Evaluate Pk, Ylm, Rn WITH release")
@btime begin B = evaluate($Pk, $X); release!(B);  end
@btime begin B = evaluate($Ylm, $X); release!(B); end
@btime begin B = evaluate($Rn, $X); release!(B);  end

@info("Gradient Pk, Ylm, Rn WITHOUT release")
@btime begin B, dB = evaluate_ed($Pk, $X);  release!(B); release!(dB); end
@btime begin B, dB = evaluate_ed($Ylm, $X); release!(B); release!(dB); end
@btime begin B, dB = evaluate_ed($Rn, $X);  release!(B); release!(dB); end

##

@btime begin B, dB = evaluate_ed($(Ylm.basis), $(X.rr)); 
             release!(B); release!(dB); end



S = ACE.SphericalHarmonics.cart2spher(X.rr)
SH = Ylm.basis
P, dP = ACE.SphericalHarmonics._evaluate_ed(SH.alp, S)          
Y = ACE.acquire!(SH.B_pool, length(SH), valtype(SH, X.rr))
dY = ACE.acquire!(SH.dB_pool, length(SH), gradtype(SH, X.rr))
_maxL = ACE.SphericalHarmonics.maxL(SH)

@btime begin P, dP = ACE.SphericalHarmonics._evaluate_ed($(SH.alp), $S); 
         release!(P); release!(dP); end 

@btime ACE.SphericalHarmonics.cYlm_ed!($Y, $dY, $_maxL, $S, $P, $dP)

@btime ACE.SphericalHarmonics.evaluate_ed!($Y, $dY, $SH, $(X.rr))

ACE.SphericalHarmonics.evaluate_ed!(Y, dY, SH, X.rr)

##

function runn(N, f, args...)
   for n = 1:N 
      Y, dY = f(args...)
      release!(Y)
      release!(dY)
   end 
end

##

runn(10, evaluate_ed, Ylm.basis, X.rr)

@profview runn(100_000, evaluate_ed, Ylm.basis, X.rr)

@btime runn(10, $evaluate_ed, $(Ylm.basis), $(X.rr))