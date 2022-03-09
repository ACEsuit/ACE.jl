

##

using ACE
using Printf, Test, LinearAlgebra, StaticArrays
using ACE: evaluate, evaluate_d, Rn1pBasis, Ylm1pBasis,
      PositionState, Product1pBasis, State, ACEConfig, 
      SymmetricBasis
using Random: shuffle
using ACEbase.Testing: dirfdtest, fdtest, print_tf, test_fio, println_slim
using ACE.OrthPolys: transformed_jacobi

##

@info("======= Testing XScal1pBasis ======= ")

maxdeg = 5
trans = ACE.Transforms.IdTransform()
P = transformed_jacobi(2*maxdeg, trans, 1.0, 0.0; pin = 0, pcut = 0) 

Bsel = ACE.SimpleSparseBasis(3, maxdeg)

##

B1p = ACE.xscal1pbasis(P, (k = 1:maxdeg, m = 0:maxdeg), :u; label = "Xkm")
ACE.init1pspec!(B1p, Bsel)
ACE.fill_rand_coeffs!(B1p, randn)

##

@info("check symbols")
println_slim(@test sort(ACE.symbols(B1p)) == sort([:k, :m]) )

@info("check `get_index`")
for ntest = 1:30 
   local idx, b 
   idx = rand(1:length(B1p))
   b = B1p.spec[idx] 
   print_tf(@test ACE.get_index(B1p, b) == idx)
end
println() 

@info("check fio")
println_slim(@test all( test_fio(B1p; warntype=false) ))


##

@info("evaluation test ")
X = State(u = rand())
B = evaluate(B1p, X)
dB = evaluate_d(B1p, X)
println_slim(@test ACE.evaluate_ed(B1p, X) == (B, dB) )


##

@info("Finite-difference tests at a few random points")
for ntest = 1:10 
   local c, F, dF 
   c = rand(length(B1p))
   F = u -> ACE.contract(c, evaluate(B1p, State(u = u)))
   dF = u -> ACE.contract(c, getproperty.(evaluate_d(B1p, State(u = u)), :u))
   print_tf(@test all( fdtest(F, dF, rand() - 0.5; verbose=false) ))
end
println() 

##

@info("check diag coeffs")
ACE.fill_diag_coeffs!(B1p, :k)
u = rand() 
J = evaluate(B1p.P, u)
B = evaluate(B1p, State(u=u))
println_slim(@test all(B[ib] == J[b.k] for (ib, b) in enumerate(B1p.spec)))

##

@info("incorporate into product basis")

Xka_u = ACE.xscal1pbasis(P, (k = 1:maxdeg, a = 0:maxdeg),:u; label = "Xka_u")
ACE.init1pspec!(Xka_u, Bsel)
Pa_v = ACE.Scal1pBasis(:v, nothing, :a, P, "Pa_v")
B1p = Xka_u * Pa_v
ACE.init1pspec!(B1p, Bsel)
println_slim(@test B1p["Xka_u"] === Xka_u)
println_slim(@test B1p["Pa_v"] === Pa_v)

rand_uv_state() = State(u = rand(), v = rand())
X = rand_uv_state()

evaluate(B1p, X)

@show length(B1p)
@show length(Xka_u)
@show length(Pa_v)
@info("sparsify")
ACE.sparsify!(B1p, ACE.get_spec(B1p))
@show length(B1p)
@show length(Xka_u)
@show length(Pa_v)

## 

@info("XScal with norm(rr) as input")
B1p = ACE.xscal1pbasis(P, (k = 1:maxdeg, m = 0:maxdeg), ACE.GetNorm{:rr}(); 
                       label = "Xkm")
ACE.init1pspec!(B1p, Bsel)
ACE.fill_rand_coeffs!(B1p, randn)

@info("check get_val, get_val_d")
X = State( rr = ACE.rand_sphere() * (0.5+rand()/2) )
println_slim(@test ACE.getval(X, B1p) ≈ norm(X.rr))
println_slim(@test ACE.getval_d(X, B1p).rr ≈ X.rr/norm(X.rr))

@info("check get_val, get_val_d")
B = evaluate(B1p, X)
dB = evaluate_d(B1p, X)
println_slim(@test ACE.evaluate_ed(B1p, X) == (B, dB))

##


@info("Finite-difference tests at a few random points")
for ntest = 1:10 
   local c, F, dF 
   c = randn(length(B1p))
   rr0 = Vector(ACE.rand_sphere() * (0.5 + rand()/0.5))
   F = rr -> ACE.contract(c, evaluate(B1p, State(rr = SVector{3}(rr))))
   dF = rr -> ACE.contract(c, getproperty.(evaluate_d(B1p, State(rr = SVector{3}(rr))), :rr)) |> Vector
   dF(rr0)
   print_tf(@test all( fdtest(F, dF, rr0; verbose=false) ))
end
println() 


##

@info("Compatibility Test")

maxdeg = 15
Bsel = ACE.SimpleSparseBasis(3, maxdeg)
trans = ACE.Transforms.IdTransform()
P = transformed_jacobi(maxdeg, trans, 1.0, 0.0; pin = 0, pcut = 0)

@info(" ... Scal1pBasis compatibility")
Bu = ACE.Scal1pBasis(:u, nothing, :k, P)
Bu_x = ACE.xscal1pbasis(P, (k = 1:maxdeg,), ACE.GetVal{:u}(), label = "Bu_x")
ACE.init1pspec!(Bu_x, Bsel)
Bu_x.coeffs[:,:] += I 

for ntest = 1:30
   X = ACE.State(u = rand())
   print_tf(@test ACE.evaluate(Bu, X) ≈ ACE.evaluate(Bu_x, X) )
end
println()


@info(" ... Rn1pBasis compatibility")
Br = ACE.Rn1pBasis(P; label = "Rn", varsym = :rr, nsym = :n)
Br_x = ACE.xscal1pbasis(P, (n = 1:maxdeg,), ACE.GetNorm{:rr}(); label = "Rn_x")
ACE.init1pspec!(Br_x, Bsel)
Br_x.coeffs[:,:] += I 

for ntest = 1:30
   rr = ACE.rand_sphere() * rand()
   X = ACE.State(rr = rr)
   print_tf(@test ACE.evaluate(Br, X) ≈ ACE.evaluate(Br_x, X) )
end
println()
