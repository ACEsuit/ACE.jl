using ACE
using ChainRulesCore


@doc raw"""
```math
\varphi(R_i) = F(\varphi_1, \dots, \varphi_P)
```
Each ``\varphi_p`` is a linear ACE.

say more?
"""
struct GfsModel{TB, TP, TEV, TF} 
   basis::TB
   c::Matrix{TP}
   evaluator::TEV
   F::TF
end

# ------------------------------------------------------------------------
#    Constructor
# ------------------------------------------------------------------------
#SymmetricBasis not defined
function GfsModel(basis, c = F.θ; 
   evaluator = :standard, F = F) 
   if evaluator == :naive 
      error("naive evaluator not yet implemented")
   elseif evaluator == :standard 
      ev = θ -> ACE.PIEvaluator(basis, θ) 
   elseif evaluator == :recursive 
      error("Recursive evaluator not yet implemented")
   else 
      error("unknown evaluator")
   end
   return GfsModel(basis, c, ev, F)
end

nparams(m::GfsModel) = size(params(m))

params(m::GfsModel) = copy(m.c)

#when called with a matrix
function set_params!(m::GfsModel, c) 
   m.c[:] .= c[:]
   #ignore for now the params in the evaluator, instead we define the eval
   #as a function and apply the params when we evaluate fully.
   #for i in 1:length(c[1,:])
   #   set_params!(m.evaluator(m.c[:,i]), m.basis, m.c[:,i])
   #end
   return m 
end

# ------------------- dispatching on the evaluators 

evaluate(m::GfsModel, X::AbstractConfiguration) = 
      m.F([evaluate(
         ACE.LinearACEModel(m.basis, m.c[:,i], m.evaluator(m.c[:,i])), X).val 
            for i in 1:length(m.c[1,:])])

# ------------------- gradients

#we expect this to return a gradient of length equal to the number of parameters
#since we keep different parameters for each ϕ we get gradients for each, so a matrix,
#imagine concatinating the columns to get one big parameter array
function grad_params(m::GfsModel, X::AbstractConfiguration) 
   #forward pass, accumulating rules
   ϕ = []
   ϕ_pullbacks = Array{Function}(undef, length(m.c[1,:]))
   for i in 1:length(m.c[1,:])
      tmp_lin = ACE.LinearACEModel(m.basis, m.c[:,i], m.evaluator(m.c[:,i]))
      a, a_pullback = evaluate(tmp_lin, X).val,  k -> k * ACE.grad_params(tmp_lin,X)
      append!(ϕ,a)
      ϕ_pullbacks[i] = a_pullback
   end
   
   b, b_pullback = Myrrule(m.F, ϕ)
   
   #backwards pass, get the gradient
   b_bar = 1 #derivative of F according to F
   a_bar = b_pullback(b_bar)
   x_bar = [ϕ_pullbacks[i](a_bar[i]) for i in 1:length(m.c[1,:])] 
   return(x_bar)
end

#could re-route to the general EVAL
struct EVAL_me{TM, TX}
   m::TM 
   X::TX
end

function (y::EVAL_me)(params)
   set_params!(y.m, params)
   return evaluate(y.m, y.X)
end

function ChainRulesCore.rrule(y::EVAL_me, params)
   set_params!(y.m, params)
   val = evaluate(y.m, y.X)
   adj = dp -> ( ChainRulesCore.NO_FIELDS, dp * @thunk(grad_params(y.m, y.X))) 
   return val, adj
end


#currently we only zygote for the params derivative. Forces, i.e config derivatives
#are not derivated through zygote.

function grad_config(m::GfsModel, X::AbstractConfiguration) 
   #forward pass, accumulating rules
   ϕ = []
   ϕ_pullbacks = Array{Function}(undef, length(m.c[1,:]))
   for i in 1:length(m.c[1,:])
      tmp_lin = ACE.LinearACEModel(m.basis, m.c[:,i], m.evaluator(m.c[:,i]))
      a, a_pullback = evaluate(tmp_lin, X).val,  k -> k * ACE.grad_config(tmp_lin,X)
      append!(ϕ,a)
      ϕ_pullbacks[i] = a_pullback
   end
   
   b, b_pullback = Myrrule(m.F, ϕ)
   
   #backwards pass, get the gradient
   b_bar = 1 #derivative of F according to F
   a_bar = b_pullback(b_bar)
   x_bar = sum([ϕ_pullbacks[i](a_bar[i]) for i in 1:length(m.c[1,:])])
   return(x_bar)
end


#This is for multiplying special types
function mul(θ, X)
   T = typeof(X[1])
   prod = Array{T}(undef, (size(θ)[1], size(X)[1]))
   for (i,t) in enumerate(θ)
      for (j,x) in enumerate(X)
         prod[i,j] = t .* x
      end
   end
   return(prod)
end

#this works for 2 ϕ ONLY, we could code one of these for each case, but so far I 
#don't see a generalization
function grad_params_config(m::GfsModel, X::AbstractConfiguration) 
   #forward pass, accumulating rules
   ϕ = []
   ϕ_pullbacks_θ = Array{Function}(undef, length(m.c[1,:]))
   ϕ_pullbacks_X = Array{Function}(undef, length(m.c[1,:]))
   ϕ_pullbacks_θX = Array{Function}(undef, length(m.c[1,:]))
   for i in 1:length(m.c[1,:])
      tmp_lin = ACE.LinearACEModel(m.basis, m.c[:,i], m.evaluator(m.c[:,i]))
      #mul since we can't multiply Svectors
      a, a_pullback_θ = evaluate(tmp_lin, X).val,  k -> mul(getproperty.(ACE.grad_params(tmp_lin,X), :val),k)
      a, a_pullback_X = a,  k -> k * ACE.grad_config(tmp_lin,X)
      a, a_pullback_θX = a,  k -> k * ACE.grad_params_config(tmp_lin,X)

      append!(ϕ,a)
      ϕ_pullbacks_θ[i] = a_pullback_θ
      ϕ_pullbacks_X[i] = a_pullback_X
      ϕ_pullbacks_θX[i] = a_pullback_θX
   end

   b, b_pullback = MyrrulePX(m.F, ϕ)

   #backwards pass, get the gradient
   b_bar = 1 #derivative of F according to F
   a_bar = b_pullback(b_bar)

   #loops over the ϕ while adding all 3 possible derivatives ddF_ϕ, dF_ϕ, dF_ϕ1ϕ2

   x_bar = [ϕ_pullbacks_θ[1]((ϕ_pullbacks_X[1](a_bar[1]))) .+ 
            ϕ_pullbacks_θX[1](a_bar[2]) .+ 
            ϕ_pullbacks_θ[1]((ϕ_pullbacks_X[2](a_bar[3]))) , 
            ϕ_pullbacks_θ[2]((ϕ_pullbacks_X[2](a_bar[4]))) + 
            ϕ_pullbacks_θX[2](a_bar[5]) + 
            ϕ_pullbacks_θ[2]((ϕ_pullbacks_X[1](a_bar[6])))]                
   
   return(x_bar)
end

#make the nonlinearities as structures. This is similar to Flux, but doesn't 
#play nice yet.

struct FinnisSinclair{ϵ}
   ϵ::ϵ
end
   
function (a::FinnisSinclair{Float64})(ϕ::Vector{Any})
   return(ϕ[1] + sqrt((a.ϵ)^2 + abs(ϕ[2])) - a.ϵ)
end
   
function (a::FinnisSinclair{Float64})(ϕ::Vector{Float64})
   return(ϕ[1] + sqrt((a.ϵ)^2 + abs(ϕ[2])) - a.ϵ)
end

struct ToyExp{} end

function (a::ToyExp)(ϕ::Vector{Any})
   return(ϕ[1]*ϕ[4] + ϕ[3]*exp(-ϕ[2]^2))
end

function (a::ToyExp)(ϕ::Vector{Float64})
   return(ϕ[1]*ϕ[4] + ϕ[3]*exp(-ϕ[2]^2))
end

function Myrrule(F::FinnisSinclair{Float64}, ϕ)
   ϕ1_θ = 1
   ϕ2_θ = (1/2)*(1/(sqrt((1/10)^2 + abs(ϕ[2]))))*(ϕ[2]/abs(ϕ[2]))
   return(F(ϕ),k -> k .* [ϕ1_θ,ϕ2_θ])
end

function MyrrulePX(F::FinnisSinclair{Float64}, ϕ)
   ddF_ϕ1 = 0
   dF_ϕ1 = 1
   ddF_ϕ2 = -(1/4)*(1/((1/10)^2 + abs(ϕ[2]))^(3/2))*(ϕ[2]/abs(ϕ[2]))
   dF_ϕ2 = (1/2)*(1/(sqrt((1/10)^2 + abs(ϕ[2]))))*(ϕ[2]/abs(ϕ[2]))
   dF_ϕ1ϕ2 = 0
   return(F(ϕ),k -> k .* [ddF_ϕ1,dF_ϕ1,dF_ϕ1ϕ2,ddF_ϕ2,dF_ϕ2,dF_ϕ1ϕ2])
end

function Myrrule(F::ToyExp, ϕ)
   dF_ϕ1 = ϕ[4]
   dF_ϕ2 = -2*ϕ[2]*ϕ[3]*exp(-ϕ[2]^2)
   dF_ϕ3 = exp(-ϕ[2]^2)
   dF_ϕ4 = ϕ[1]
   return(F(ϕ),k -> k .* [dF_ϕ1,dF_ϕ2,dF_ϕ3,dF_ϕ4])
end

using JuLIP

function E_θ(m,at,vref,θ)
   E = 0.0
   nlist = neighbourlist(at, cutoff(vref))
   for i = 1:length(at)
       Js, Rs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
       Rs=ACEConfig([EuclideanVectorState(Rs[j],"r") for j in 1:length(Rs)])
       Ei = ACE.Models.EVAL_me(m,Rs)(θ)
       E=Ei
   end
   return E
end

#could re-route to the general EVAL
struct ENERGY_me{TM, TA, TV}
   m::TM 
   at::TA
   vref::TV
end

function (y::ENERGY_me)(params)
   set_params!(y.m, params)
   E = 0.0
   nlist = neighbourlist(y.at, cutoff(y.vref))
   for i = 1:length(y.at)
       Js, Rs, Zs = JuLIP.Potentials.neigsz(nlist, y.at, i); z0 = y.at.Z[i]
       Rs=ACEConfig([EuclideanVectorState(Rs[j],"r") for j in 1:length(Rs)])
       Ei = ACE.Models.EVAL_me(y.m,Rs)(params)
       E += Ei
   end
   return E
end

function ChainRulesCore.rrule(y::ENERGY_me, params)
   set_params!(y.m, params)
   val = y(params)
   function adj(dp)
      nlist = neighbourlist(y.at, cutoff(y.vref))
      Js, Rs, Zs = JuLIP.Potentials.neigsz(nlist, y.at, 1); z0 = y.at.Z[1]
      Rs=ACEConfig([EuclideanVectorState(Rs[j],"r") for j in 1:length(Rs)])
      tmp = grad_params(y.m, Rs)
      for i = 2:length(y.at)
         Js, Rs, Zs = JuLIP.Potentials.neigsz(nlist, y.at, i); z0 = y.at.Z[i]
         Rs=ACEConfig([EuclideanVectorState(Rs[j],"r") for j in 1:length(Rs)])
         tmp += grad_params(y.m, Rs)
     end
     return ( ChainRulesCore.NO_FIELDS, dp * tmp) 
   end

   return val, adj
end