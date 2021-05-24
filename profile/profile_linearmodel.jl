

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

function runn(N, f, args...)
   t = f(args...)
   for n = 2:N
     t = f(args...)
   end 
   t
end

##

@info("Time evaluate")
@btime evaluate($standard, $cfg)

@info("Time grad_config")
@btime ACE.grad_config($standard, $cfg)

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