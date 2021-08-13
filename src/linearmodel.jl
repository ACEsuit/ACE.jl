#
# Draft ACEModel Interface: 
#
# * A model incorporates both the parameterisation and the parameters. 
# * E.g. a LinearACEModel would know about the basis and the coefficients
# * We can then perform the following operations: 
#    - evaluate the model at a given configuration, with given parameters
#    - take gradient w.r.t. the configuration 
#    - take gradient w.r.t. the parameters 
#
# An advantage will be that we can do the parameter reduction trick and then 
# use a fast evaluator to obtain the gradient w.r.t. configuration which is 
# very expensive if we naively take derivatives w.r.t. the basis and then 
# apply parameters. (this is not implemented below and will be the next step)
# Another advantage is that model, and full model construction are all stored 
# together for future inspection.

struct LinearACEModel{TB, TP, TEV} <: AbstractACEModel 
   basis::TB
   c::Vector{TP}
   evaluator::TEV   
   # grad_params_pool::VectorPool{TP}
end 

struct NaiveEvaluator end 

function LinearACEModel(basis::SymmetricBasis, c = zeros(length(basis)); 
               evaluator = :standard) 
   if evaluator == :naive 
      ev = NaiveEvaluator()
   elseif evaluator == :standard 
      ev = ProductEvaluator(basis, c) 
   elseif evaluator == :recursive 
      error("Recursive evaluator not yet implemented")
   else 
      error("unknown evaluator")
   end
   return LinearACEModel(basis, c, ev)
end

# LinearACEModel(basis::SymmetricBasis, c::Vector, evaluator) = 
#          LinearACEModel(basis, c, ev, VectorPool{eltype(c)})

# ------- parameter wrangling 

nparams(m::LinearACEModel) = length(m.c)

params(m::LinearACEModel) = copy(m.c)

function set_params!(m::LinearACEModel, c) 
   m.c[:] .= c
   set_params!(m.evaluator, m.basis, c)
   return m 
end

set_params!(::NaiveEvaluator, args...) = nothing 

# ------------------- FIO

==(V1::LinearACEModel, V2::LinearACEModel) = 
      _allfieldsequal(V1, V2)

write_dict(V::LinearACEModel) = 
      Dict( "__id__" => "ACE_LinearACEModel", 
             "basis" => write_dict(V.basis), 
                 "c" => write_dict(V.c), 
         "evaluator" => write_dict(V.evaluator) )

function read_dict(::Val{:ACE_LinearACEModel}, D::Dict) 
   basis = read_dict(D["basis"])
   c = read_dict(D["c"])
   # special evaluator version of the read_dict 
   evaluator = read_dict(Val(Symbol(D["evaluator"]["__id__"])), 
                         D["evaluator"], basis, c)
   return LinearACEModel(basis, c, evaluator)
end

write_dict(ev::NaiveEvaluator) = 
      Dict("__id__" => "ACE_NaiveEvaluator" )

read_dict(::Val{:ACE_NaiveEvaluator}, D::Dict, args...) = 
      NaiveEvaluator()




# ------- managing temporaries 

# TODO: consider providing a generic object pool / array pool 
# acquire!(m.grad_cfg_pool, length(cfg), gradtype(m.basis, X))
acquire_grad_config!(m::LinearACEModel, cfg::AbstractConfiguration) = 
   Matrix{gradtype(m.basis, cfg)}(undef, length(cfg), length(m.c[1]))
#Vector{gradtype(m.basis, cfg)}(undef, length(cfg))

release_grad_config!(m::LinearACEModel, g) = nothing 
      #release!(m.grad_cfg_pool, g)

acquire_grad_params!(m::LinearACEModel, args...) = 
      alloc_B!(m.basis, m.c, args...)

#This probably needs adjusting since we are now alocating a matrix
function alloc_B!(basis::ACEBasis, c::AbstractVector , args...) 
   VT = valtype(basis, args...)
   return Matrix{VT}(undef, length(basis), length(c[1]))
end

# function acquire_B!(basis::ACEBasis, c::AbstractVector , args...) 
#    VT = valtype(basis, args...)
#    if hasproperty(basis, :B_pool)
#       return acquire!(basis.B_pool, length(basis), VT)
#    end 
#    return Vector{VT}(undef, length(basis))
# end

release_grad_params!(m::LinearACEModel, g) = 
      release_B!(m.basis, g)

# ------------------- dispatching on the evaluators 

evaluate(m::LinearACEModel, X::AbstractConfiguration) = 
      evaluate(m::LinearACEModel, m.evaluator, X::AbstractConfiguration)

grad_config(m::LinearACEModel, X::AbstractConfiguration) = 
      grad_config!(acquire_grad_config!(m, X), m, X)

grad_config!(g, m::LinearACEModel, X::AbstractConfiguration) = 
      grad_config!(g, m, m.evaluator, X) 

grad_params(m::LinearACEModel, cfg::AbstractConfiguration) = 
      grad_params!(acquire_grad_params!(m, cfg), m, cfg)

function grad_params!(g, m::LinearACEModel, cfg::AbstractConfiguration) 
   for i in 1:size(g,2)
      g[:,i] = evaluate!(g[:,i], m.basis, cfg)
   end 
   return g 
end

#lazy fix for now. We need to think about allocations and weather we want these
#derivatives as a vector of matrices, a tensor, or even if we want them at all.
function grad_params_config(m::LinearACEModel, cfg::AbstractConfiguration) 
   dB = acquire_dB!(m.basis, cfg)
   return [grad_params_config!(dB, m, cfg) for i = 1:length(m.c[1])]
end

# ∂_params ∂_config V
# this function should likely never be used in production, but could be 
# useful for testing
grad_params_config!(dB, m::LinearACEModel, cfg::AbstractConfiguration)  = 
      evaluate_d!(dB, m.basis, cfg) 

# TODO: fix terminology, bring in linear with the _rrule_.... thing 
adjoint_EVAL_D(m::LinearACEModel, cfg::AbstractConfiguration, w) = 
      adjoint_EVAL_D(m, m.evaluator, cfg, w)


# ------------------- implementation of naive evaluator 
#  this is only intended for testing, as it uses the naive evaluation of  
#  the symmetric basis, rather than the conversion to the AA basis

function evaluate(m::LinearACEModel, ::NaiveEvaluator, cfg::AbstractConfiguration)  
   B = acquire_B!(m.basis, cfg)
   evaluate!(B, m.basis, cfg)
   val = sum(prod, zip(m.c, B))
   release_B!(m.basis, B) 
   return val 
end 

function grad_config!(g, m::LinearACEModel, ::NaiveEvaluator, 
                     cfg::AbstractConfiguration)
   dB = acquire_dB!(m.basis, cfg) 
   evaluate_d!(dB, m.basis, cfg) 
   fill!(g, zero(eltype(g)))
   for ix = 1:length(cfg), ib = 1:length(m.basis)
      g[ix] += m.c[ib] * dB[ib, ix]
   end
   release_dB!(m.basis, dB)
   return g 
end

function adjoint_EVAL_D(m::LinearACEModel, ::NaiveEvaluator, 
                        X::AbstractConfiguration, w) 
   dB = grad_params_config(m, X)
   g = zeros(size(dB, 1))
   for i = 1:length(g), j = 1:size(dB, 2)
      g[i] += dot(dB[i, j], w[j])
   end
   release_dB!(m.basis, dB)
   return g
end


# ------------------- dispatching the rrule 

import ChainRules: rrule, @thunk, NoTangent, @not_implemented

function rrule(::typeof(evaluate), m::LinearACEModel, env::AbstractConfiguration)
   val = evaluate(m, env)
   pullback = dv -> (NoTangent(), @thunk(grad_params(m, env)), 
                                @thunk(grad_config(m, env)))
   return val, pullback
end
