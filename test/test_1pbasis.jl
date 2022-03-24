

##

using ACE
using Printf, Test, LinearAlgebra, StaticArrays
using ACE: evaluate, evaluate_d, Rn1pBasis, Ylm1pBasis,
      PositionState, Product1pBasis, getlabel, get_spec, 
      State, rand_radial, rand_sphere, Scal1pBasis, 
      valtype, gradtype, acquire_B!, acquire_dB!
using Random: shuffle
using ACEbase.Testing: dirfdtest, fdtest, print_tf, test_fio, println_slim

##


@info "Build a 1p basis from scratch"

maxdeg = 12
maxL = 5
r0 = 1.0
rcut = 3.0
maxorder = 3
Bsel = SimpleSparseBasis(maxorder, maxdeg)

trans = PolyTransform(1, r0)   # r -> x = 1/r^2
J = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)   #  J_n(x) * (x - xcut)^pcut
Rn = Rn1pBasis(J)
Ylm = Ylm1pBasis(maxL)
Pk = Scal1pBasis(:u, nothing, :k, J)
A_nlm = Product1pBasis( (Rn, Ylm) )
ACE.init1pspec!(A_nlm, Bsel)
A_nlmk = Product1pBasis( (Rn, Ylm, Pk) )
ACE.init1pspec!(A_nlmk, Bsel)

nX = 10
Xs = [ State(rr = rand_radial(J) * rand_sphere() ) for _=1:nX ]
cfg = ACEConfig(Xs)

A = evaluate(A_nlm, Xs)

@info("test against manual summation")
A1 = sum( evaluate(A_nlm, X) for X in Xs )
println_slim(@test A1 ≈ A)

@info("test permutation invariance")
for ntest = 1:30
   print_tf(@test A ≈ evaluate(A_nlm, ACEConfig(shuffle(Xs))))
end
println()

## 
@info("test access via labels")
println_slim(@test(getlabel(Ylm) == "Ylm"))
println_slim(@test(getlabel(Rn) == "Rn"))
println_slim(@test(getlabel(Pk) == "Pk"))
println_slim(@test(A_nlm["Ylm"] === Ylm))
println_slim(@test(A_nlm["Rn"] === Rn))
println_slim(@test(A_nlmk["Pk"] === Pk))

##

@info("Test FIO")
for _B in (J, Rn, Ylm, Pk, A_nlm, A_nlmk)
   print(string(Base.typename(typeof(_B)))[10:end-1], " - ", getlabel(_B), " : ")
   println_slim((@test(all(test_fio(_B)))))
end

##

@info("Testing gradients for several 1p basis components")
for basis in (Pk, Rn, Ylm, A_nlm, A_nlmk)
   local X 
   @info(" .... $(basis)")
   _randX() = State( rr = rand_radial(J) * rand_sphere(), u = rand_radial(J) )
   X = _randX()
   B = acquire_B!(basis, X)
   dB = acquire_dB!(basis, X)
   B1 = acquire_B!(basis, X)
   dB1 = acquire_dB!(basis, X)
   # println_slim(@test (typeof(dY) == eltype(Ylm.dB_pool.arrays[Base.Threads.threadid()])))
   ACE.evaluate!(B, basis, X)
   ACE.evaluate_d!(dB, basis, X)
   ACE.evaluate_ed!(B1, dB1, basis, X) 
   println_slim(@test (evaluate(basis, X) ≈ B))
   println_slim(@test (evaluate_d(basis, X) ≈ dB))
   println_slim(@test ((B ≈ B1) && (dB ≈ dB1)) )

   # this could be moved into ACE proper ... 
   z_contract(dx::DState, u::DState) = 
         sum( ACE.contract(getproperty(dx, sym), getproperty(u, sym))
              for sym in ACE._syms(dx) )

   for ntest = 1:30
      X = _randX() 
      U = DState(_randX())
      B = evaluate(basis, X)
      c = randn(length(B))
      F = t -> sum( evaluate(basis, X + t * U) .* c )
      dF = t -> sum( c .* z_contract.( evaluate_d(basis, X + t * U), Ref(U) ) )
      
      tf = fdtest(F, dF, 0.0; verbose=false)
      print_tf(@test all(tf))
   end
   println() 

end



##


@info("Product 1p basis test on configurations")

for basis in (A_nlm, A_nlmk)   
   @info(" .... $(basis)")
   local Xs, cfg, A1, nX
   nX = 5
   Xs = [ State(rr = rand_radial(J) * rand_sphere(), 
                u = rand_radial(J) ) for _=1:nX ]
   cfg = ACEConfig(Xs)
   A1 = ACE.acquire_B!(basis, cfg)
   ACE.evaluate!(A1, basis, cfg)
   A2 = ACE.acquire_B!(basis, cfg)
   dA = ACE.acquire_dB!(basis, cfg)
   ACE.evaluate_ed!(A2, dA, basis, cfg)
   println_slim(@test A1 ≈ A2)
   println_slim(@test( evaluate_d(basis, Xs) ≈ dA ))

   for ntest = 1:30
      nX = 5
      _randX() = State( rr = rand_radial(J) * rand_sphere(), u = rand_radial(J) )
      Xs = [_randX() for _=1:5] 
      Us = [DState(_randX()) for _=1:5]
      c = rand(length(basis))
      F = t -> sum( evaluate(basis, Xs + t * Us) .* c )
      dF = t -> begin 
         dB = evaluate_d(basis, Xs + t * Us)
         c_dB = sum(c .* dB, dims=1)[:]
         ACE.contract(c_dB, Us)
      end
      print_tf(@test fdtest(F, dF, 0.0; verbose=false))
   end
   println()

end