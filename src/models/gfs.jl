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
   return(ϕ[1] + exp(-ϕ[2]^2))
end

function (a::ToyExp)(ϕ::Vector{Float64})
   return(ϕ[1] + exp(-ϕ[2]^2))
end

function Myrrule(F::FinnisSinclair{Float64}, ϕ)
   ϕ1_θ = 1
   ϕ2_θ = (1/2)*(1/(sqrt((1/10)^2 + abs(ϕ[2]))))*(ϕ[2]/abs(ϕ[2]))
   return(F(ϕ),k -> k .* [ϕ1_θ,ϕ2_θ])
end

function Myrrule(F::ToyExp, ϕ)
   ϕ1_θ = 1
   ϕ2_θ = -2*ϕ[2]*exp(-ϕ[2]^2)
   return(F(ϕ),k -> k .* [ϕ1_θ,ϕ2_θ])
end