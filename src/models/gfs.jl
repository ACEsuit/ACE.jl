using ACE

struct NaiveEvaluator end 

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
      ev = θ -> NaiveEvaluator()
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

set_params!(::NaiveEvaluator, args...) = nothing 

# ------------------- dispatching on the evaluators 

evaluate(m::GfsModel, X::AbstractConfiguration) = 
      m.F([evaluate(
         ACE.LinearACEModel(m.basis, m.c[:,i], m.evaluator(m.c[:,i])), X).val 
            for i in 1:length(m.c[1,:])])

# ------------------- gradients

function grad_params(m::GfsModel, X::AbstractConfiguration) 
   #forward pass, accumulating rules
   ρ = []
   ρ_pullbacks = Array{Function}(undef, length(m.c[1,:]))
   for i in 1:length(m.c[1,:])
      tmp_lin = ACE.LinearACEModel(m.basis, m.c[:,i], m.evaluator(m.c[:,i]))
      a, a_pullback = evaluate(tmp_lin, X).val,  k -> k * ACE.grad_params(tmp_lin,X)
      append!(ρ,a)
      ρ_pullbacks[i] = a_pullback
   end
   
   b, b_pullback = Myrrule(m.F, ρ)
   
   #backwards pass, get the gradient
   b_bar = 1 #derivative of F according to F
   a_bar = b_pullback(b_bar)
   x_bar = [ρ_pullbacks[i](a_bar[i]) for i in 1:length(m.c[1,:])] 
   return(x_bar)
end

function grad_config(m::GfsModel, X::AbstractConfiguration) 
   #forward pass, accumulating rules
   ρ = []
   ρ_pullbacks = Array{Function}(undef, length(m.c[1,:]))
   for i in 1:length(m.c[1,:])
      tmp_lin = ACE.LinearACEModel(m.basis, m.c[:,i], m.evaluator(m.c[:,i]))
      a, a_pullback = evaluate(tmp_lin, X).val,  k -> k * ACE.grad_config(tmp_lin,X)
      append!(ρ,a)
      ρ_pullbacks[i] = a_pullback
   end
   
   b, b_pullback = Myrrule(m.F, ρ)
   
   #backwards pass, get the gradient
   b_bar = 1 #derivative of F according to F
   a_bar = b_pullback(b_bar)
   x_bar = [ρ_pullbacks[i](a_bar[i]) for i in 1:length(m.c[1,:])] 
   return(x_bar)
end

# function grad_params_config(m::GfsModel, X::AbstractConfiguration) 
#    #forward pass, accumulating rules
#    ρ = []
#    ρ_pullbacks_θ = Array{Function}(undef, length(m.c[1,:]))
#    ρ_pullbacks_X = Array{Function}(undef, length(m.c[1,:]))
#    ρ_pullbacks_θX = Array{Function}(undef, length(m.c[1,:]))
#    for i in 1:length(m.c[1,:])
#       tmp_lin = ACE.LinearACEModel(m.basis, m.c[:,i], m.evaluator(m.c[:,i]))
#       a, a_pullback_θ = evaluate(tmp_lin, X).val,  k -> k * ACE.grad_params(tmp_lin,X)
#       a, a_pullback_X = a,  k -> k * ACE.grad_config(tmp_lin,X)
#       a, a_pullback_θX = a,  k -> k * ACE.grad_params_config(tmp_lin,X)
#       append!(ρ,a)
#       ρ_pullbacks_θ[i] = a_pullback_θ
#       ρ_pullbacks_X[i] = a_pullback_X
#       ρ_pullbacks_θX[i] = a_pullback_θX
#    end
   
#    b, b_pullback = Myrrule(m.F, ρ)
   
#    #backwards pass, get the gradient
#    b_bar = 1 #derivative of F according to F
#    a_bar = b_pullback(b_bar)
#    x_bar = [ρ_pullbacks[i](a_bar[i]) for i in 1:length(m.c[1,:])] 
#    return(x_bar)
# end


abstract type NonLinearity end
struct FinnisSinclair{ϵ} <: NonLinearity
   ϵ::ϵ
end

function (a::FinnisSinclair{Float64})(ρ::Vector{Any})
   return(ρ[1] + sqrt((a.ϵ)^2 + abs(ρ[2])) - a.ϵ)
end

function (a::FinnisSinclair{Float64})(ρ::Vector{Float64})
   return(ρ[1] + sqrt((a.ϵ)^2 + abs(ρ[2])) - a.ϵ)
end

struct ToyExp{} <: NonLinearity  end

function (a::ToyExp)(ρ::Vector{Any})
   return(ρ[1] + exp(-ρ[2]^2))
end

function (a::ToyExp)(ρ::Vector{Float64})
   return(ρ[1] + exp(-ρ[2]^2))
end

function Myrrule(F::FinnisSinclair{Float64}, ρ)
   ρ1_θ = 1
   ρ2_θ = (1/2)*(1/(sqrt((1/10)^2 + abs(ρ[2]))))*(ρ[2]/abs(ρ[2]))
   return(F(ρ),k -> k .* [ρ1_θ,ρ2_θ])
end

function Myrrule(F::ToyExp, ρ)
   ρ1_θ = 1
   ρ2_θ = -2*ρ[2]*exp(-ρ[2]^2)
   return(F(ρ),k -> k .* [ρ1_θ,ρ2_θ])
end



# * constructors: 
#    basis, F, maybe c -> GFinnisSinclairACE
# * parameter wrangling: 
#      nparams, params, set_params!
# * evaluation codes: start with 
#      - evaluate 

# design interface for F, e.g., 
#        evaluate(F, phi)
#        grad_config(F, phi)
#        grad_params(F, phi)

