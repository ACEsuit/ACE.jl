
##

using Yota
using ACE, ACEbase, Zygote, ChainRules, BenchmarkTools, StaticArrays
using Printf, Test, LinearAlgebra, ACE.Testing, Random
using ACE: evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis
using ACEbase.Testing: fdtest
import ChainRulesCore: rrule, NoTangent


## [1] 


maxdeg = 20
trans = ACE.Transforms.IdTransform()
P = transformed_jacobi(maxdeg, trans, 1.0, 0.0)

function f1_1(P, x)
   a = 1 ./ (1:length(P))
   b = 0.3 ./ (1:length(P)).^2
   return dot(a, evaluate(P, x)) * dot(b, evaluate_d(P, x))
end

x0 = rand()
Zygote.refresh()
Zygote.gradient(f1_1, P, x0)
fdtest(x -> f1_1(P, x), x -> Zygote.gradient(f1_1, P, x)[2], x0)

@btime f1_1($P, $x0)
@btime Zygote.gradient($f1_1, $P, $x0)


## 


# here is an example where Zygote differentiation is god-awful .. 

function f1_2(P, x)
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
Zygote.gradient(f1_2, P, x0)
fdtest(x -> f1_2(P, x), x -> Zygote.gradient(f1_2, P, x)[2], x0)

@btime f1_2($P, $x0)
@btime Zygote.gradient($f1_2, $P, $x0)


## [g] chainrules for State and DState 

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

##

# w = SVector{3}(rand(3))
# x, pb = rrule(getproperty, X2, :rr)
# @btime $pb($w)
# @code_warntype(pb(w))


## [3] differentiate an Rn basis 

Rn = ACE.Rn1pBasis(P)
r0 = ACE.rand_radial(Rn) * ACE.Random.rand_sphere()
X = PositionState(r0)
evaluate(Rn, X)
evaluate_d(Rn, X)

function f3_1(Rn, X)
   a = 1 ./ (1:length(Rn))
   Rn_ = evaluate(Rn, X)
   return dot(a, Rn_) 
end

@info("testing AD for Rn with f3_1 :")
Zygote.refresh()
g = Zygote.gradient(f3_1, Rn, X)[2]

x = X.rr
fdtest( x -> f3_1(Rn, PositionState(x)), 
      x -> Vector(Zygote.gradient(f3_1, Rn, PositionState(x))[2].rr), 
      Vector(X.rr) )

@info("""differentiate f3_1, which only involves `evaluate` gives decent 
      performance: 
      ~ factor 30 without custom adjoint 
      ~ factor 3.5 with custom adjoint """)
@btime f3_1($Rn, $X)
@btime Zygote.gradient($f3_1, $Rn, $X)


##  another including also evaluate_d

f3_2_Rn_in(Rs) = mapreduce(r -> 1 / (1+sum(abs2, r)), +, Rs) 

_rrule_f3_2_Rn_in(Rs, W::Number) = 
      [ (-2 * W * r) / (1+sum(abs2, r))^2  for r in Rs ]
           
ChainRules.rrule(::typeof(f3_2_Rn_in), Rs) =
      f3_2_Rn_in(Rs), Ws -> ( NoTangent(), _rrule_f3_2_Rn_in(Rs, Ws) )

function f3_2(Rn, X)
   dRn_ = getproperty.(evaluate_d(Rn, X), :rr)
   return f3_2_Rn_in(dRn_)
   # mapreduce(dr -> 1/(1 + sum(abs2, rr)), +, dRn_)
   # return sum( 1/(1 + norm(dr.rr))^2 for dr in dRn_ )
end

f3_2(Rn, X)

@info("testing AD for Rn with f3_2 :")
Zygote.refresh()
g = Zygote.gradient(f3_2, Rn, X)[2]

x = X.rr
fdtest( x -> f3_2(Rn, PositionState(x)), 
      x -> Vector(Zygote.gradient(f3_2, Rn, PositionState(x))[2].rr), 
      Vector(X.rr) )

@info(""" timeing of Zygote.gradient for f3_2 - this involves 
         evaluate_d and some semi-complicated nonlinear function. 
         The timing here is awful: ~ factor 1000 without custom adjoint
                                   ~ factor 40 with custom adjoint. """)
@btime f3_2($Rn, $X)
@btime Zygote.gradient($f3_2, $Rn, $X)
@info("""
   but this doesn't seem to be caused by the rrule implementation 
   this is actually quite fast - this should at least in principle 
   allow for fast adjoints: factor evaluate_d vs rrule is ~ 2.13 """)
w = [ ACE.DState(rr = rand(SVector{3, Float64})) for _ = 1:length(Rn) ]
w = [ ACE.DState( rr = rand(SVector{3, Float64}) ) for _ = 1:length(Rn) ]
dRn = evaluate_d(Rn.R, norm(X.rr))
@btime ACE.evaluate_d($Rn, $X)
@btime ACE._rrule_evaluate_d($Rn, $X, $w, $dRn)


##  [4] Ylm basis 

@info("Testing adjoints for the Ylm basis")

@info("""First step: remind ourselves of the performance of allocating and 
         non-allocating evaluation and gradients.""")
Ylm = ACE.Ylm1pBasis(6)
B = ACE.alloc_B(Ylm, X)
tmp = ACE.alloc_temp(Ylm, X)
@btime evaluate($Ylm, $X)
@btime ACE.evaluate!($B, $Ylm, $X)
@btime ACE.evaluate!($B, $tmp, $Ylm, $X)
dB = ACE.alloc_dB(Ylm, X)
tmpd = ACE.alloc_temp_d(Ylm, X)
@btime evaluate_d($Ylm, $X)
@btime ACE.evaluate_d!($dB, $tmpd, $Ylm, $X)
dY = zeros(SVector{3, ComplexF64}, length(Ylm))
@btime ACE.evaluate_d!($dY, $(Ylm.SH), $(X.rr)) 
@btime ACE.evaluate_d!($dB, $Ylm, $X) 

##

function f4_1(Ylm, X)
   a = 1 ./ (1:length(Ylm))
   Ylm_ = evaluate(Ylm, X)
   f = sum(a .* Ylm_) 
   return real(f) * cos(imag(f))
end

f4_1(Ylm, X)

Zygote.refresh()
Zygote.gradient(f4_1, Ylm, X)[2]

x0 = Vector(X.rr)
fdtest( x -> f4_1( Ylm, ACE.State(rr = SVector{3}(x)) ), 
        x -> Zygote.gradient(f4_1, Ylm, ACE.State(rr = SVector{3}(x)))[2].rr |> Vector, 
        x0 )

@info("""timing for f4_1: not too terrible 
             f ~ 500ns, df ~ 3us => factor 6""")
@btime f4_1($Ylm, $X)
@btime Zygote.gradient($f4_1, $Ylm, $X);


## [5] Scalar 1p Basis (the one for the invariant features....)
#   looking ok ...

@info("AD tests for the scalar 1p basis")

using ACE: State 
maxdeg = 10 
r0 = 1.0 
rcut = 3.0 
bscal = ACE.scal1pbasis(:x, :k, maxdeg, PolyTransform(1, r0), rcut)

ACE.evaluate(bscal, State(x = 1.2))
ACE.evaluate_d(bscal, State(x = 1.0))

X = State(x=1.1)
f5_1(basis::ACE.Scal1pBasis, X) = sum( (exp ∘ cos).(evaluate(basis, X)) )

f5_1(bscal, X)
Zygote.gradient(f5_1, bscal, X)

fdtest(x -> f5_1(bscal, State(x = x)), 
       x -> Zygote.gradient(f5_1, bscal, State(x=x))[2].x, 
       X.x)


@info("  Timings: not great, gradient(f5_1) / f5_1 ~ factor 20")
@btime f5_1($bscal, $X)
@btime Zygote.gradient($f5_1, $bscal, $X)

##

X = State(x=1.1)
f5_2(basis::ACE.Scal1pBasis, X) = 
      sum( (exp ∘ cos).( getproperty.(evaluate_d(basis, X), :x) ) )

f5_2(bscal, X)
Zygote.gradient(f5_2, bscal, X)

fdtest(x -> f5_2(bscal, State(x = x)), 
       x -> Zygote.gradient(f5_2, bscal, State(x=x))[2].x, 
       X.x)

@info("""  Second test differentiating evaluate_d; 
           Ratio is a bit better gradient(f5_2) / f5_2 ~ factor 14
           but this might be just because f5_2 has extra allocations itself""")
@btime f5_2($bscal, $X)
@btime Zygote.gradient($f5_2, $bscal, $X)


## [6] then a product basis as a function of a single argument  

B1p = Rn * Ylm * bscal
ACE.init1pspec!(B1p; maxdeg = maxdeg, Deg = ACE.NaiveTotalDegree())

X = State( rr = rand(SVector{3, Float64}), x = rand() )
evaluate(B1p, X)
evaluate_d(B1p, X)

##

function f6_1(B1p, X)
   B = evaluate(B1p, X)
   a = 1 ./ (1:length(B1p))
   b = 1 ./ (1:length(B1p)).^2
   return sum( (real.(B) .* a).^2 + cos.(imag.(B)) .* b )^2
#    return real( sum(a .* B) )
#    return sum(b .* imag.(B) + a .* real.(B))
end

Zygote.refresh()
f6_1(B1p, X)
_df1 = Zygote.gradient(f6_1, B1p, X)[2]

##

_x2X = x -> State( rr = SVector{3}(x[1:3]), x = x[4] )
_X2x = X -> [ X.rr; [X.x] ]
x0 = _X2x(X)
F = x -> f6_1(B1p, _x2X(x))
dF = x -> _X2x(Zygote.gradient(f6_1, B1p, _x2X(x))[2])
fdtest(F, dF, x0)

##

@info("""timing test for rrule for evaluate-Product1pBasis
            evaluate: ~ 35 us 
            Zygote.gradient: ~ 537 us
            rrule: ~ 360 us 
      => factor 10; this code is full of type instability and 
      can surely be improved!
      """)
@btime f6_1($B1p, $X)
@btime Zygote.gradient($f6_1, $B1p, $X);
w = evaluate(B1p, X)
BB = ACE._evaluate_bases(B1p, X)
@btime ACE._rrule_evaluate($B1p, $X, $w, $BB)
@code_warntype ACE._rrule_evaluate(B1p, X, w, BB)


## [7] density projection (argument is a vector)

Rn
_rndX() = State( rr = (Rn.R.rl + rand() * (Rn.R.ru - Rn.R.rl)) * ACE.Random.rand_sphere(), 
                 x = (bscal.P.rl + rand() * (bscal.P.ru - bscal.P.rl)) )
cfg = ACEConfig( [_rndX() for _=1:10 ])

function f7_1(B1p, cfg)
      B = evaluate(B1p, cfg)
      a = 1 ./ (1:length(B1p))
      b = 1 ./ (1:length(B1p)).^2
      return sum( (real.(B) .* a).^2 + cos.(imag.(B)) .* b )^2
end

##


f7_1(B1p, cfg)
Zygote.refresh()
Zygote.gradient(f7_1, B1p, cfg)

## fd test 

_dotuu = (u1, u2) -> dot(u1.rr, u2.rr) + dot(u1.x, u2.x)
Us = DACEConfig( [ _rndX() for _=1:10 ] ) 
cfg_t = t -> cfg + t * Us 
F = t -> f7_1(B1p, cfg_t(t))
dF = t -> sum(  _dotuu(u, dx)  for (u, dx) in zip(Us, 
                        Zygote.gradient(f7_1, B1p, cfg_t(t))[2] ) )
fdtest(F, dF, 0.0)

## timing 

@info("Timing of 1p-basis on cfg")

@info("   evaluate vs evaluate!")
A = ACE.alloc_B(B1p, cfg)
tmp = ACE.alloc_temp(B1p, cfg)
@btime evaluate($B1p, $cfg)
@btime ACE.evaluate!($A, $B1p, $cfg)
@btime ACE.evaluate!($A, $tmp, $B1p, $cfg)


@info("""   evaluate_ed! vs rrule
            note that evaluate_ed! is currently hyper-not optimised!!!""")

dA = ACE.alloc_dB(B1p, cfg);
tmpd = ACE.alloc_temp_d(B1p, cfg);
W = rand(ComplexF64, length(B1p))
print("    evaluate_ed!: "); @btime ACE.evaluate_ed!($A, $dA, $tmpd, $B1p, $cfg);
print(" _rrule_evaluate: "); @btime ACE._rrule_evaluate($B1p, $cfg, $W);

@info("""   f7_1 vs AD-gradient: 
            ca factor 65; needs a lot of improvement""")
@btime f7_1($B1p, $cfg);
@btime Zygote.gradient( $f7_1, $B1p, $cfg); 


## PIbasis




## SymmetricBasis


## Linear ACE Model


## energies and forces



# @info("Basic test of LinearACEModel construction and evaluation")

# # construct the 1p-basis
# D = NaiveTotalDegree()
# maxdeg = 6
# ord = 3
# B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

# # generate a configuration
# nX = 10
# Xs = rand(EuclideanVectorState, B1p.bases[1], nX)
# cfg = ACEConfig(Xs)

# fltype(B1p)

# ##
