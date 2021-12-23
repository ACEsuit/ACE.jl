

using ACE, ACEbase
using Printf, LinearAlgebra, ACE.Testing, Random
using ACE: evaluate, evaluate_d, SymmetricBasis, SimpleSparseBasis, PIBasis, 
           PositionState
using ACEbase.Testing: fdtest
using Profile, ProfileView, TimerOutputs, BenchmarkTools

##

# construct the 1p-basis
maxdeg = 14
ord = 4
Bsel = SimpleSparseBasis(ord, maxdeg)

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, Bsel=Bsel)

# generate a configuration
nX = 10
Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
cfg = ACEConfig(Xs)

φ = ACE.Invariant()
basis = SymmetricBasis(φ, B1p, Bsel)
@show length(basis)

BB = evaluate(basis, cfg)
c = rand(length(BB)) .- 0.5

standard = ACE.LinearACEModel(basis, c, evaluator = :standard)

##

@info("Time evaluate incl allocation")
@btime evaluate($standard, $cfg)

@info("Time grad_config with and without allocation")
g = ACE.acquire_grad_config!(standard, cfg)
@btime ACE.grad_config($standard, $cfg)
@btime ACE.grad_config!($g, $standard, $cfg)

##

@info("Time grad_params with and without allocation")
g = ACE.acquire_grad_params!(standard, cfg)
@btime ACE.grad_params($standard, $cfg)
@btime ACE.grad_params!($g, $standard, $cfg)

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