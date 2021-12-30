

using ACE, ACEbase, StaticArrays
using Printf, LinearAlgebra, ACE.Testing, Random
using ACE: evaluate, evaluate_d, SymmetricBasis, SimpleSparseBasis, PIBasis, 
           PositionState
using ACEbase.Testing: fdtest
using Profile, ProfileSVG, TimerOutputs, BenchmarkTools

##

# construct the 1p-basis
maxdeg = 14
ord = 4
wL = 1.5
Bsel = SparseBasis(; maxorder = ord, p = 1, default_maxdeg = maxdeg, 
                     weight = Dict(:n => 1.0, :l => wL))

B1p = ACE.Utils.RnYlm_1pbasis(maxdeg = maxdeg, maxL = ceil(Int, maxdeg/wL), 
                              Bsel = Bsel)

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
(@btime evaluate($standard, $cfg))

##

@info("Time grad_config with and without allocation")
@info("This looks like a factor 4.5 of evaluate, so probably more we can do")
g = ACE.acquire_grad_config!(standard, cfg)
(@btime ACE.grad_config($standard, $cfg))
(@btime ACE.grad_config!($g, $standard, $cfg))

##

@info("grad_config vs rrule_evaluate")
g1 = ACE._rrule_evaluate(ACE._One(), standard, cfg)
g2 = ACE.grad_config(standard, cfg)
@show g1 ≈ g2  
@btime ACE.grad_config($standard, $cfg)
@btime ACE._rrule_evaluate(ACE._One(), $standard, $cfg)

##

@info("Time grad_params with and without allocation")
@info("a little surprising we dont get closer to factor 1?")
g = ACE.acquire_grad_params!(standard, cfg)
(@btime ACE.grad_params($standard, $cfg))
(@btime ACE.grad_params!($g, $standard, $cfg))

##

@info("adjoint_EVAL_D - single property")
@info("This may be the single-most important function, and it looks fantastic!")
#create a random input emulating the pullback input
w = [ACE.DState(rr = randn(SVector{3, Float64})) for j in 1:length(cfg)]
@btime ACE.adjoint_EVAL_D($standard, $cfg, $w)

##

@info("Multi-property")

c_m = randn(SVector{2, Float64}, length(basis))
model2 = ACE.LinearACEModel(basis, c_m, evaluator = :standard)

@info(" - evaluate")
@btime evaluate($model2, $cfg)
@info(" - grad_params")
@btime ACE.grad_params($model2, $cfg)
@info(" - grad_config")
@btime ACE.grad_config($model2, $cfg)
w20 = randn(SVector{2, Float64})
@info(" - _rrule_evaluate")
@btime ACE._rrule_evaluate($w20, $model2, $cfg)

@info(" - adjoint_EVAL_D")
_w2() = SVector(ACE.DState(rr = randn(SVector{3, Float64})), 
                ACE.DState(rr = randn(SVector{3, Float64})))
w2 = [ _w2() for j in 1:length(cfg)]
ACE.adjoint_EVAL_D(model2, cfg, w2)
@btime ACE.adjoint_EVAL_D($model2, $cfg, $w2)

##

# function runn(N, f, args...)
#    t = f(args...)
#    for n = 2:N
#      t = f(args...)
#    end 
#    t
# end

# ##

# Profile.clear()
# @profile runn(10000, ACE.adjoint_EVAL_D, standard, cfg, w)
# Profile.print()

# ##

# ProfileSVG.view()


##

# # make sure to add suitable @timeit macros to the module
# @info("Detailed benchmarking of grad_config")
# reset_timer!()
# @timeit "grad_config" runn(10, ACE.grad_config, standard, cfg);
# print_timer()