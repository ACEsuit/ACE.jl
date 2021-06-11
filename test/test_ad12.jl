
##


using ACE, ACEbase, Zygote, ChainRules, BenchmarkTools, StaticArrays
using Printf, Test, LinearAlgebra, ACE.Testing, Random
using ACE: evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis
using ACEbase.Testing: fdtest
import ChainRulesCore: rrule 

##


maxdeg = 20
trans = ACE.Transforms.IdTransform()
P = transformed_jacobi(maxdeg, trans, 1.0, 0.0)

function f1(P, x)
   a = 1 ./ (1:length(P))
   b = 0.3 ./ (1:length(P)).^2
   return dot(a, evaluate(P, x)) * dot(b, evaluate_d(P, x))
end

x0 = rand()
Zygote.refresh()
Zygote.gradient(f1, P, x0)
fdtest(x -> f1(P, x), x -> Zygote.gradient(f1, P, x)[2], x0)

@btime f1($P, $x0)
@btime Zygote.gradient($f1, $P, $x0)


## 


# here is an example where Zygote differentiation is god-awful .. 

function f2(P, x)
   a = 1 ./ (1:length(P))
   b = 0.3 ./ (1:length(P)).^2
   B = evaluate(P, x)
   dB = evaluate_d(P, x)
   # soso performance (factor 16)
   return sum( a .* B .* exp.(- b .* dB) )
   # awful performance!! (factor 400)
   # return sum( a[i] * B[i] * exp(-b[i] * dB[i]^2)  
   #             for i = 1:length(P))
end

Zygote.refresh()
Zygote.gradient(f2, P, x0)
fdtest(x -> f2(P, x), x -> Zygote.gradient(f2, P, x)[2], x0)

@btime f2($P, $x0)
@btime Zygote.gradient($f2, $P, $x0)


## chainrules for State and DState 

@info("Checking rrules for State and DState")

X1 = PositionState(rand(SVector{3, Float64}))
g1(X1) = sum(abs2, X1.rr)
Zygote.refresh()
println( @test( Zygote.gradient(g1, X1)[1].rr ≈ 2 * X1.rr ))

X2 = ACE.State(rr = rand(SVector{3, Float64}), u = rand())
_g = Zygote.gradient(g1, X2)[1]
println( @test( (_g.rr ≈ 2 * X2.rr && _g.u == 0) ))

X3 = ACE.DState(rr = rand(SVector{3, Float64}), u = rand())
_g = Zygote.gradient(g1, X3)[1]
println( @test( (_g.rr ≈ 2 * X3.rr && _g.u == 0) ))


g2(X) = sum(abs2, X.rr) * exp(- 0.3 * X.u)
_g = Zygote.gradient(g2, X2)[1]
println( @test( (_g.rr ≈ 2 * X2.rr * exp(-0.3*X2.u)) && (_g.u ≈ - 0.3 * g2(X2)) ))



## differentiate an Rn basis 

Rn = ACE.Rn1pBasis(P)
r0 = ACE.rand_radial(Rn) * ACE.Random.rand_sphere()
X = PositionState(r0)
evaluate(Rn, X)
evaluate_d(Rn, X)

function f1(Rn, X)
   a = 1 ./ (1:length(Rn))
   Rn_ = evaluate(Rn, X)
   return dot(a, Rn_) 
end

@info("testing AD for Rn with f1 :")
Zygote.refresh()
g = Zygote.gradient(f1, Rn, X)[2]

x = X.rr
fdtest( x -> f1(Rn, PositionState(x)), 
      x -> Vector(Zygote.gradient(f1, Rn, PositionState(x))[2].rr), 
      Vector(X.rr) )

@info("""differentiate f1, which only involves `evaluate` gives decent 
      performance: 
      ~ factor 30 without custom adjoint 
      ~ factor 3.5 with custom adjoint """)
@btime f1($Rn, $X)
@btime Zygote.gradient($f1, $Rn, $X)


##  another including also evaluate_d

function f2(Rn, X)
   dRn_ = evaluate_d(Rn, X)
   mapreduce(dr -> 1/(1 + sum(abs2, dr.rr)), +, dRn_)
   # return sum( 1/(1 + norm(dr.rr))^2 for dr in dRn_ )
end

@info("testing AD for Rn with f2 :")
Zygote.refresh()
g = Zygote.gradient(f2, Rn, X)[2]

x = X.rr
fdtest( x -> f2(Rn, PositionState(x)), 
      x -> Vector(Zygote.gradient(f2, Rn, PositionState(x))[2].rr), 
      Vector(X.rr) )

@info(""" timeing of Zygote.gradient for f2 - this involves 
         evaluate_d and some semi-complicated nonlinear function. 
         The timing here is awful: ~ factor 1000""")
@btime f2($Rn, $X)
@btime Zygote.gradient($f2, $Rn, $X)
@info("""
   but this doesn't seem to be caused by the rrule implementation 
   this is actually quite fast - this should at least in principle 
   allow for fast adjoints: factor evaluate_d vs rrule is ~ 2.13 """)
w = [ ACE.DState(rr = rand(SVector{3, Float64})) for _ = 1:length(Rn) ]
w = [ ACE.DState( rr = rand(SVector{3, Float64}) ) for _ = 1:length(Rn) ]
dRn = evaluate_d(Rn.R, norm(X.rr))
@btime ACE.evaluate_d($Rn, $X)
@btime ACE._rrule_evaluate_d($Rn, $X, $w, $dRn)


## Ylm basis 




## then a product basis as a function of a single argument 



## density projection (argument is a vector)


## PIbasis


## SymmetricBasis


## Linear ACE Model


## energies and forces



@info("Basic test of LinearACEModel construction and evaluation")

# construct the 1p-basis
D = NaiveTotalDegree()
maxdeg = 6
ord = 3
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

# generate a configuration
nX = 10
Xs = rand(EuclideanVectorState, B1p.bases[1], nX)
cfg = ACEConfig(Xs)

fltype(B1p)

##
