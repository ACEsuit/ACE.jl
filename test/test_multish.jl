


##

using ACE
using StaticArrays, Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis, 
           State, O3 
using ACE.Random: rand_rot, rand_refl
using ACEbase.Testing: fdtest
using ACE.Testing: __TestSVec

rands3nrm() = ( rr = randn(SVector{3, Float64}); rr / norm(rr) )
rands3(rl, ru) = (rl + rand() * (ru-rl)) * rands3nrm()



## FIRST TEST (l, m) vs custom (lr, ms) 

# construct the 1p-basis
D = NaiveTotalDegree()
maxdeg = 5
ord = 3

B1p_r = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D, 
                                  varsym = :rr, idxsyms = (:nr, :lr, :mr))
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D, )
@show ACE.symbols(B1p_r)
@show ACE.indexrange(B1p_r)
println(@test all( ACE.indexrange(B1p_r)[symr] == ACE.indexrange(B1p)[sym]
                   for (symr, sym) in ((:nr, :n), (:lr, :l), (:mr, :m)) ) )
φ = ACE.Invariant()
basis = SymmetricBasis(φ, B1p, O3(), ord, maxdeg; Deg = D)
basis_r = SymmetricBasis(φ, B1p_r, O3(:lr, :mr), ord, maxdeg; Deg = D)

ru = basis.pibasis.basis1p.bases[1].R.ru 
rl = basis.pibasis.basis1p.bases[1].R.rl 

for ntest = 1:30 
   cfg = ACEConfig( [ State(rr = rands3(rl, ru)) for _=1:10 ] )
   print_tf(@test evaluate(basis, cfg) ≈ evaluate(basis_r, cfg))
end 

## 

@info("Construct a basis for (rr, ss) without spin-orbit coupling")

maxdeg = 6

X = State( rr = rands3(rl, ru), ss = rands3nrm() )
MagState = typeof(X)
Base.rand(::Type{MagState}) = MagState( rr = rands3(rl, ru), ss = rands3nrm() )
Base.rand(::Type{MagState}, N::Integer) = [ rand(MagState) for _ = 1:N ]

B1p_r = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D, 
                                  varsym = :rr, idxsyms = (:nr, :lr, :mr))
B1p_s = ACE.Ylm1pBasis(maxdeg; varsym = :ss, lsym = :ls, msym = :ms)
B1p = B1p_r * B1p_s
@show ACE.symbols(B1p)
@show ACE.indexrange(B1p)


ACE.init1pspec!(B1p; maxdeg=maxdeg, Deg = D)
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
basis = SymmetricBasis(φ, B1p, O3(:lr, :mr), ord, maxdeg; Deg = D)

@info("Check for consistenty of r-rotation")
for ntest = 1:30
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
println(@test (toterr_rs > 1))
println(@test (toterr_rr > 1))

## 

ord = 4
maxdeg = 7
basis = SymmetricBasis(φ, B1p, O3(:lr, :mr) ⊗ O3(:ls, :ms), ord, maxdeg; Deg = D)
# basis = SymmetricBasis(φ, B1p, O3(:lr, :mr), ord, maxdeg; Deg = D)

# for ntest = 1:30
cfg = ACEConfig(rand(MagState, nX))
Qr = ACE.Random.rand_rot() * ACE.Random.rand_refl()
Qs = ACE.Random.rand_rot() * ACE.Random.rand_refl()
cfg_rs = ACEConfig( shuffle( [MagState(rr = Qr * X.rr, ss = Qs * X.ss) for X in cfg] ) )
B = evaluate(basis, cfg)
B_rs = evaluate(basis, cfg_rs)
@show norm(B - B_rs, Inf)

spec = ACE.get_spec(basis)

# [ evaluate(basis, cfg)  evaluate(basis, cfg_sym)  spec ] |> display 

##
ctr = 0
ctr_first = 0
for i = 1:length(basis) 
   if !(B[i] ≈ B_rs[i])
      print("i = $i : ")
      display( spec[i] )
      ctr += 1 
      if ctr_first == 0; ctr_first = i; end 
      if ctr == 10; break; end 
   end
end
ctr_first

##


display(spec[ [200, 205, 210, 211, 213] ])

