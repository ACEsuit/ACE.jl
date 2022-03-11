

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
B1p = Product1pBasis( (Rn, Ylm) )
ACE.init1pspec!(B1p, Bsel)


nX = 10
Xs = [ State(rr = rand_radial(J) * rand_sphere() ) for _=1:nX ]
cfg = ACEConfig(Xs)

A = evaluate(B1p, Xs)

@info("test against manual summation")
A1 = sum( evaluate(B1p, X) for X in Xs )
println_slim(@test A1 ≈ A)

@info("test permutation invariance")
for ntest = 1:30
   print_tf(@test A ≈ evaluate(B1p, ACEConfig(shuffle(Xs))))
end
println()

## 
@info("test access via labels")
println_slim(@test(getlabel(Ylm) == "Ylm"))
println_slim(@test(getlabel(Rn) == "Rn"))
println_slim(@test(B1p["Ylm"] == Ylm))
println_slim(@test(B1p["Rn"] == Rn))

##

@info("Test FIO")
for _B in (J, Rn, Ylm, Pk, B1p)
   print(string(Base.typename(typeof(_B)))[10:end-1], " - ", getlabel(_B), " : ")
   println_slim((@test(all(test_fio(_B)))))
end

##

@info("Testing gradients for several 1p basis components")
for basis in (Pk, Rn, Ylm)
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

@info("Product basis evaluate_ed! tests")

A1 = ACE.acquire_B!(B1p, cfg)
ACE.evaluate!(A1, B1p, cfg)
A2 = ACE.acquire_B!(B1p, cfg)
dA = ACE.acquire_dB!(B1p, cfg)
ACE.evaluate_ed!(A2, dA, B1p, cfg)
println_slim(@test A1 ≈ A2)

println_slim(@test( evaluate_d(B1p, Xs) ≈ dA ))

##
@info("Product basis gradient test")

for ntest = 1:30
   x0 = randn(3)
   c = rand(length(B1p))
   F = x -> sum(ACE.evaluate(B1p, _vec2X(x)) .* c)
   dF = x -> sum(ACE.evaluate_d(B1p, ACEConfig([_vec2X(x)])) .* c).rr |> Vector
   print_tf(@test fdtest(F, dF, x0; verbose=false))
end
println()
##

