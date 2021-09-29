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
nX = 54
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
   Ftemp = Zygote.gradient(x -> sum(energyModel(θ, x)), cfg)[1]
   return(abs2(sum(energyModel(θ, cfg))) + sum([sum(Ftemp[i].rr) for i in 1:length(Ftemp)]))
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

#same function just doesn't enforce x to be an AbstractArray
function fdtestMatrix(F, dF, x; h0 = 1.0, verbose=true)
   errors = Float64[]
   E = F(x)
   dE = dF(x)
   # loop through finite-difference step-lengths
   verbose && @printf("---------|----------- \n")
   verbose && @printf("    h    | error \n")
   verbose && @printf("---------|----------- \n")
   for p = 2:11
      h = 0.1^p
      dEh = copy(dE)
      for n = 1:length(dE)
         x[n] += h
         dEh[n] = (F(x) - E) / h
         x[n] -= h
      end
      push!(errors, norm(dE - dEh, Inf))
      verbose && @printf(" %1.1e | %4.2e  \n", h, errors[end])
   end
   verbose && @printf("---------|----------- \n")
   if minimum(errors) <= 1e-3 * maximum(errors)
      verbose && println("passed")
      return true
   else
      @warn("""It seems the finite-difference test has failed, which indicates
      that there is an inconsistency between the function and gradient
      evaluation. Please double-check this manually / visually. (It is
      also possible that the function being tested is poorly scaled.)""")
      return false
   end
end

##
#the actual testing

for ntest = 1:30
    c = randn(np, length(basis))
    F = t ->  loss(matrix2svector(c + t))
    dF = t -> svector2matrix(Zygote.gradient(loss, matrix2svector(c))[1])
    print_tf(@test fdtestMatrix(F, dF, zeros(np, length(basis)), verbose=false))
end
println()