using ACE

#struct NaiveEvaluator end 

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
      ev = θ -> ACE.PIEvaluator(basis, θ) #naive not implemented
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

#set_params!(::NaiveEvaluator, args...) = nothing 

# ------------------- dispatching on the evaluators 

evaluate(m::GfsModel, X::AbstractConfiguration) = 
      m.F([evaluate(
         ACE.LinearACEModel(m.basis, m.c[:,i], m.evaluator(m.c[:,i])), X).val 
            for i in 1:length(m.c[1,:])])

# ------------------- gradients

#we expect this to return a gradient of length equal to the number of parameters
#since we keep different parameters for each ϕ we get gradients for each, so a matrix
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

function grad_params_config(m::GfsModel, X::AbstractConfiguration) 
   #forward pass, accumulating rules
   ϕ = []
   ϕ_pullbacks_θ = Array{Function}(undef, length(m.c[1,:]))
   ϕ_pullbacks_X = Array{Function}(undef, length(m.c[1,:]))
   ϕ_pullbacks_θX = Array{Function}(undef, length(m.c[1,:]))
   for i in 1:length(m.c[1,:])
      tmp_lin = ACE.LinearACEModel(m.basis, m.c[:,i], m.evaluator(m.c[:,i]))
      a, a_pullback_θ = evaluate(tmp_lin, X).val,  k -> k * ACE.grad_params(tmp_lin,X)
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
  
   #loops over the ϕ while adding all 3 possible derivatives dX, dθ, dXθ
   x_bar = []
   i = 1
   j = 1
   while(i <= length(m.c[1,:]))
      append!(x_bar,ϕ_pullbacks_X[i](a_bar[j]) + ϕ_pullbacks_θ[i](a_bar[j+1]) + ϕ_pullbacks_θX[i](a_bar[j+2]))
      i += 1
      j += 3
   end
   return(x_bar)
end

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

#this is not the way to do it in the future
function MyrrulePX(F::FinnisSinclair{Float64}, ϕ)
   ϕ1_Xθ = 1
   ϕ2_θ = -(1/4)*(1/((1/10)^2 + abs(ϕ[2]))^(3/2))*(ϕ[2]/abs(ϕ[2]))
   ϕ2_Xθ = (1/2)*(1/(sqrt((1/10)^2 + abs(ϕ[2]))))*(ϕ[2]/abs(ϕ[2]))
   #current convention is ϕ1_X,ϕ1_θ,ϕ1_Xθ,ϕ2_X,...ϕN_Xθ
   return(F(ϕ),k -> k .* [0,0,ϕ1_Xθ,0,ϕ2_θ,ϕ2_Xθ])
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

