
##


using ACE, ACEbase, Zygote, ChainRules, BenchmarkTools
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

@btime f1($P, $x)
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

@btime f2($P, $x)
@btime Zygote.gradient($f2, $P, $x0)


## differentiate an Rn basis 

# Rn = ACE.Rn1pBasis(P)
# r0 = ACE.rand_radial(Rn) * ACE.Random.rand_sphere()
# evaluate(Rn, r0)

# then a product basis as a function of a single argument 



# density projection (argument is a vector)



# PIbasis



# SymmetricBasis




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
