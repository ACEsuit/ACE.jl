

using ACE, ACEbase
using Printf, LinearAlgebra, ACE.Testing, Random
using ACE: evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis
using ACEbase.Testing: fdtest
using Profile, ProfileView, TimerOutputs, BenchmarkTools

# construct the 1p-basis
D = NaiveTotalDegree()
maxdeg = 14
ord = 4

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D)

# generate a configuration
nX = 10
Xs = rand(EuclideanVectorState, B1p.bases[1], nX)
cfg = ACEConfig(Xs)

φ = ACE.Invariant()
pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
basis = SymmetricBasis(pibasis, φ)
@show length(basis)

BB = evaluate(basis, cfg)
c = rand(length(BB)) .- 0.5

standard = ACE.LinearACEModel(basis, c, evaluator = :standard)

##

@info("Time evaluate incl allocation")
@btime evaluate($standard, $cfg)

@info("Time grad_config incl allocation")
@btime ACE.grad_config($standard, $cfg)

##

@info("Time evaluate excl allocation")
tmp = ACE.alloc_temp(standard)
@btime ACE.evaluate!($tmp, $standard, $cfg)

@info("Time grad_config excl allocation")
g = ACE.alloc_grad_config(standard, cfg)
tmp = ACE.alloc_temp_d(standard, length(cfg))
@btime ACE.grad_config!($g, $tmp, $standard, $cfg)


##

function runn(N, f, args...)
   t = f(args...)
   for n = 2:N
     t = f(args...)
   end 
   t
end

##

# Profile.clear()
# @profile runn(100, ACE.grad_config, standard, cfg);
# Profile.print()

# ##

# # make sure to add suitable @timeit macros to the module
# @info("Detailed benchmarking of grad_config")
# reset_timer!()
# @timeit "grad_config" runn(10, ACE.grad_config, standard, cfg);
# print_timer()