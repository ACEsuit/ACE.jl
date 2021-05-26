@testset "Experimental AD"  begin

##


using ACE, ACEbase, Zygote, ChainRules
using Printf, Test, LinearAlgebra, ACE.Testing, Random
using ACE: evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis
using ACEbase.Testing: fdtest
using ACE: EVAL, EVAL_D
import ChainRulesCore: rrule 
##

@info("Basic test of LinearACEModel construction and evaluation")

# construct the 1p-basis
D = NaiveTotalDegree()
maxdeg = 8
ord = 4

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

# generate a configuration
nX = 10
Xs = rand(EuclideanVectorState, B1p.bases[1], nX)
cfg = ACEConfig(Xs)

φ = ACE.Invariant()
pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
basis = SymmetricBasis(pibasis, φ)

BB = evaluate(basis, cfg)
c = rand(length(BB)) .- 0.5
model = ACE.LinearACEModel(basis, c, evaluator = :standard)
@show length(basis)

##

e = EVAL(model, cfg)
e(c)
Zygote.gradient(e, c)[1]
ACEbase.Testing.fdtest(e, c -> Zygote.gradient(e, c)[1], c)

##

norm2(x::AbstractVector) = sum(abs2, x)

de = EVAL_D(model, cfg)
de(c)
fd = c -> sum(norm2, de(c))
# fd = c -> norm2(de(c)[1])
fd(c)
Zygote.gradient(fd, c)[1]
ACEbase.Testing.fdtest(fd, c -> Zygote.gradient(fd, c)[1], c)

##


cfgs = [ ACEConfig(rand(EuclideanVectorState, B1p.bases[1], nX))
         for _ = 1:10 ]
loss = params -> sum( ( EVAL(model, cfg)(params) 
                         + sum(norm2, EVAL_D(model, cfg)(params)) )
                      for cfg in cfgs ) / length(cfgs) / nX
loss(c)
Zygote.gradient(loss, c)[1]
ACEbase.Testing.fdtest(loss, c -> Zygote.gradient(loss, c)[1], c)

##

naive = ACE.LinearACEModel(basis, c, evaluator = :naive)
loss_naive = params -> sum( ( EVAL(naive, cfg)(params) 
                           + sum(norm2, EVAL_D(naive, cfg)(params)) )
                           for cfg in cfgs ) / length(cfgs) / nX
loss_naive(c)
Zygote.gradient(loss_naive, c)[1]

##

@time loss(c)
@time loss_naive(c)
@time Zygote.gradient(loss, c)[1]
@time Zygote.gradient(loss_naive, c)[1]

##

end   