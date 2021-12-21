


##

using ACE
using StaticArrays, Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, SymmetricBasis, PIBasis, 
           State, O3 
using ACE.Random: rand_rot, rand_refl
using ACEbase.Testing: fdtest, println_slim 
using ACE.Testing: __TestSVec

rands3nrm() = ( rr = randn(SVector{3, Float64}); rr / norm(rr) )
rands3(rl, ru) = (rl + rand() * (ru-rl)) * rands3nrm()



## FIRST TEST (l, m) vs custom (lr, ms) 

# construct the 1p-basis
maxdeg = 5
ord = 3
Bsel = SimpleSparseBasis(ord, maxdeg)

B1p_r = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, 
                                  varsym = :rr, idxsyms = (:nr, :lr, :mr))
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg,  )
@show ACE.symbols(B1p_r)
@show ACE.indexrange(B1p_r)
println_slim(@test all( ACE.indexrange(B1p_r)[symr] == ACE.indexrange(B1p)[sym]
                   for (symr, sym) in ((:nr, :n), (:lr, :l), (:mr, :m)) ) )
φ = ACE.Invariant()
basis = SymmetricBasis(φ, B1p, O3(), Bsel)
basis_r = SymmetricBasis(φ, B1p_r, O3(:lr, :mr), Bsel)

ru = basis.pibasis.basis1p.bases[1].R.ru 
rl = basis.pibasis.basis1p.bases[1].R.rl 

for ntest = 1:30 
   local cfg 
   cfg = ACEConfig( [ State(rr = rands3(rl, ru)) for _=1:10 ] )
   print_tf(@test evaluate(basis, cfg) ≈ evaluate(basis_r, cfg))
end 
println()

## 

@info("Construct a basis for (rr, ss) without spin-orbit coupling")

maxdeg = 6

X = State( rr = rands3(rl, ru), ss = rands3nrm() )
MagState = typeof(X)
Base.rand(::Type{MagState}) = MagState( rr = rands3(rl, ru), ss = rands3nrm() )
Base.rand(::Type{MagState}, N::Integer) = [ rand(MagState) for _ = 1:N ]

B1p_r = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, 
                                  varsym = :rr, idxsyms = (:nr, :lr, :mr))
B1p_s = ACE.Ylm1pBasis(maxdeg; varsym = :ss, lsym = :ls, msym = :ms)
B1p = B1p_r * B1p_s
@show ACE.symbols(B1p)
@show ACE.indexrange(B1p)


ACE.init1pspec!(B1p, Bsel)
length(B1p)
spec = ACE.get_spec(B1p)

# generate a configuration
nX = 10
Xs = rand(MagState, nX)
cfg = ACEConfig(Xs)
evaluate(B1p, Xs[1])
evaluate(B1p, cfg)


##

@info("Symmetrize only w.r.t. rr but leave ss.")

ord = 4
maxdeg = 7
basis = SymmetricBasis(φ, B1p, O3(:lr, :mr), Bsel)

@info("Check for consistenty of r-rotation")
for ntest = 1:30
   local cfg 
   cfg = ACEConfig(rand(MagState, nX))
   Qr = ACE.Random.rand_rot() * ACE.Random.rand_refl()
   cfg_sym = ACEConfig( shuffle( [MagState(rr = Qr * X.rr, ss = X.ss) for X in cfg] ) )   
   print_tf(@test( evaluate(basis, cfg) ≈ evaluate(basis, cfg_sym) )) 
end
println()

@info("Check for inconsistency of s-rotation")
toterr_rs = 0.0 
toterr_rr = 0.0 
for ntest = 1:30
   global toterr_rs, toterr_rr
   local cfg 
   cfg = ACEConfig(rand(MagState, nX))
   Qr = ACE.Random.rand_rot() * ACE.Random.rand_refl()
   Qs = ACE.Random.rand_rot() * ACE.Random.rand_refl()
   cfg_rs = ACEConfig( shuffle( [MagState(rr = Qr * X.rr, ss = Qs * X.ss) for X in cfg] ) )
   cfg_rr = ACEConfig( shuffle( [MagState(rr = Qr * X.rr, ss = Qr * X.ss) for X in cfg] ) )
   toterr_rs += norm( evaluate(basis, cfg) - evaluate(basis, cfg_rs), Inf )
   toterr_rr += norm( evaluate(basis, cfg) - evaluate(basis, cfg_rr), Inf )
   print(".")
end
println()
println_slim(@test (toterr_rs > 1))
println_slim(@test (toterr_rr > 1))

## 

@info("Now test a basis with O(3) ⊗ O(3) symmetry")
ord = 3
maxdeg = 7
Bsel = SimpleSparseBasis(ord, maxdeg)
basis = SymmetricBasis(φ, B1p, O3(:lr, :mr) ⊗ O3(:ls, :ms), Bsel)
@show length(basis) 

for ntest = 1:30
   local cfg 
   cfg = ACEConfig(rand(MagState, nX))
   Qr = ACE.Random.rand_rot() * ACE.Random.rand_refl()
   Qs = ACE.Random.rand_rot() * ACE.Random.rand_refl()
   cfg_rs = ACEConfig( shuffle( [MagState(rr = Qr * X.rr, ss = Qs * X.ss) for X in cfg] ) )
   B = evaluate(basis, cfg)
   B_rs = evaluate(basis, cfg_rs)
   print_tf(@test(B ≈ B_rs))
end
println()

##