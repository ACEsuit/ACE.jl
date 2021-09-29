using LinearAlgebra: length
using ACE, ACEbase, Test, ACE.Testing
using ACE: evaluate, SymmetricBasis, PIBasis, O3, State
using StaticArrays
using ChainRules
import ChainRulesCore: rrule, NoTangent
using Zygote
using Printf, LinearAlgebra #for the fdtestMatrix

##

@info("loss function test")

# construct the basis
maxdeg = 6
ord = 3
Bsel = SimpleSparseBasis(ord, maxdeg)
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
φ = ACE.Invariant()
basis = SymmetricBasis(φ, B1p, O3(), Bsel)

# # generate a random configuration
nX = 10
cfg = ACEConfig([State(rr = rand(SVector{3, Float64})) for _ in 1:length(nX)])

#initialize the model
np = 2
c_m = rand(SVector{np,Float64}, length(basis))
model = ACE.LinearACEModel(basis, c_m, evaluator = :standard)

##

#define the site energy and site forces calculators
#for this code we define "model" globaly to make the code easier
#to understand (no derivatives according to model objects).
#However, for the full implementation it should be there. 

#a hack to implement .. meaning iterating twice over something
f(x) = getproperty.(x, :val)

#calculates the site energy
function energyModel(θ, cfg)
   ACE.set_params!(model, θ)
   return getproperty.(evaluate(model, cfg), :val)
end

#calculates the adjoing/pullback 
function adj(dp, θ, cfg)
   ACE.set_params!(model, θ)
   gp = f.(ACE.grad_params(model, cfg))
   for i = 1:length(gp) 
      gp[i] = gp[i] .* dp
   end

   g_cfg = ACE.grad_config(model, cfg) #TODO multiply by dp
   # @show typeof(g_cfg)
   # @show size(g_cfg) 

   # for j = 1:length(dp)
   #    g_cfg[j] *= dp[j] 
   # end

   return (NoTangent(), gp, g_cfg) #d(dp), d(θ), d(cfg)
end

function ChainRules.rrule(::typeof(energyModel), θ, cfg)
   return energyModel(θ, cfg), dp -> adj(dp, θ, cfg) 
end

#chainrule for derivative of forces according to parameters
function ChainRules.rrule(::typeof(adj), dp, θ, cfg)
   function secondAdj(dq)
      grad = ACE.adjoint_EVAL_D(model, cfg, dq[3])
      return (NoTangent(), NoTangent(), grad, NoTangent()) #only keep dF^2/dθd(cfg)
   end
   return(adj(dp, θ, cfg), secondAdj)
end

#simple loss function with sum over properties and over forces
function loss(θ)
   props = energyModel(θ, cfg)
   # FS = props -> sum( [ 0.77^n * (1 + props[n]^2)^(1/n) for n = 1:length(props) ] )
   FS = props -> sum( [0.77^n for n = 1:length(props)] .* props )
   Ftemp = Zygote.gradient(x -> FS( energyModel(θ, x) ), cfg)[1]
   floss = f -> sum(abs2, f.rr)
   return(abs2(FS(props)) + sum(floss, Ftemp))
end

# g = Zygote.gradient(loss, c_m)[1] sample on how to get the gradient

##

#functions for testing. basically handling SVectors and testing multiple
#properties

function svector2matrix(sv)
   M = zeros(length(sv[1]), length(sv))
   for i in 1:length(sv)
      M[:,i] = sv[i]
   end
   return M
end

function matrix2svector(M)
   sv = [SVector{size(M)[1]}(M[:,i]) for i in 1:size(M)[2]]
   return sv
end


##

c = randn(np * length(basis))
F = c -> loss(matrix2svector(reshape(c, np, length(basis))))
dF = c -> svector2matrix(Zygote.gradient(loss, matrix2svector(reshape(c, np, length(basis))))[1])[:]

println(@test ACEbase.Testing.fdtest(F, dF, c, verbose=true))
