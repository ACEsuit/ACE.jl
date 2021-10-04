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

"""
`struct LinearACEModel`: linear model for symmetric properties in terms of 
a `SymmetricBasis`. 

The typical way to construct a linear model is to first construct a basis 
`basis`, some default coefficients `c` and then call 
```julia
model = LinearACEModel(basis, c)
```

### Multiple properties 

If `c::Vector{<: Number}` then the output of the model will be the property 
encoded in the basis. But one can also use a single basis to produce 
multiple properties (with different coefficients). This can be achieved by 
simply supplying `c::Vector{SVector{N, T}}` where `N` will then be the 
number of properties. 
"""
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
   acquire_grad_config!(m, cfg, m.c)

acquire_grad_config!(m::LinearACEModel, cfg::AbstractConfiguration, c::AbstractVector{<: SVector}) =
   Matrix{gradtype(m.basis, cfg)}(undef, length(cfg), length(m.c[1]))

acquire_grad_config!(m::LinearACEModel, cfg::AbstractConfiguration, c::AbstractVector{<: Number}) =
   Vector{gradtype(m.basis, cfg)}(undef, length(cfg))

release_grad_config!(m::LinearACEModel, g) = nothing 
      #release!(m.grad_cfg_pool, g)

acquire_grad_params!(m::LinearACEModel, args...) = 
      acquire_B!(m.basis, args...)

release_grad_params!(m::LinearACEModel, g) = 
      release_B!(m.basis, g)


# TODO: somehow it feels wrong that valtype should depend on c. Here the reason 
#       is that c lives in the model and not in the basis. We should trace
#       back how this occured and if possible remove these two methods. 
#       maybe they can be replaced with "private" methods, then I'd be more 
#       comfortable. 
function ACEbase.valtype(basis::ACEBasis, cfg::AbstractConfiguration, c::AbstractVector{<: SVector})
   return SVector{length(c[1]), valtype(basis, zero(eltype(cfg)))}
end

#calls the regular valtype
ACEbase.valtype(basis::ACEBasis, cfg::AbstractConfiguration, c::AbstractVector{<: Number}) =
      valtype(basis, cfg)

ACEbase.valtype(model::LinearACEModel, cfg) = 
      ACEbase.valtype(model.basis, cfg, model.c)

# ACE.gradtype(model, cfg)      

# ------------------- dispatching on the evaluators 

evaluate(m::LinearACEModel, X::AbstractConfiguration) = 
      evaluate(m::LinearACEModel, m.evaluator, X::AbstractConfiguration)

grad_config(m::LinearACEModel, X::AbstractConfiguration) = 
      grad_config!(acquire_grad_config!(m, X), m, X)

grad_config!(g, m::LinearACEModel, X::AbstractConfiguration) = 
      grad_config!(g, m, m.evaluator, X) 

grad_params(m::LinearACEModel, cfg::AbstractConfiguration) = 
      grad_params!(acquire_grad_params!(m, cfg, m.c), m, cfg)

function grad_params!(g, m::LinearACEModel, cfg::AbstractConfiguration) 
   evaluate!(g, m.basis, cfg) 
   return g 
end

# currently doesn't work with multiple properties
function grad_params_config(m::LinearACEModel, cfg::AbstractConfiguration) 
   dB = acquire_dB!(m.basis, cfg)
   return grad_params_config!(dB, m, cfg)
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


# ------------------- an rrule for evaluating a linear model

import ChainRules: rrule, @thunk, NoTangent, @not_implemented

# NOTE: there are lots of rrule implementations in `evaluate.jl`, which seem
#       to be doing the same thing and are clashing with this. So for now we 
#       just get something up and running but we must return to this and 
#       clean up all those different versions of the same thing. 

# adj_evaluate is defined as a separate function so we can rrule it again (see below)
# should be called _rrule_evaluate, but that would clash with the 
# crap that's in evaluator.jl -> needs some cleanup 
function adj_evaluate(dp, model::ACE.LinearACEModel, cfg)
   # dp = getproperty.(dp_, :val)
   gp_ = ACE.grad_params(model, cfg)
   gp = [ val.(a .* dp) for a in gp_ ]
   g_cfg = ACE._rrule_evaluate(dp, model, cfg) # rrule for cfg only...
   return NoTangent(), gp, g_cfg
end

# this is monkey-patching the rotten rrule inside of ACE
# ... and should replace that rrule. ALso introduce thunks to prevent 
#     evaluating more than we need.
function ChainRules.rrule(::typeof(evaluate), model::ACE.LinearACEModel, cfg::AbstractConfiguration)
   return evaluate(model, cfg), 
          dp -> adj_evaluate(dp, model, cfg)
end

# rrule for the rrule ... this enables mixed second derivatives of the form 
#   D^2 * / D p Dcfg. 
# the code double-checks that indeed only those derivatives are needed! 
function ChainRules.rrule(::typeof(adj_evaluate), dp, model::ACE.LinearACEModel, cfg)

   # dp should be a vector of the same length as the number of properties

   function _second_adj(dq_)
      # adj = (NoTangent(), gp1, g_cfg) 
      # here we assume that only g_cfg was used, which means that 
      # dq_[3] = force-like vector and dq_[2] == NoTangent() 
      @assert dq_[1] == dq_[2] == ZeroTangent()
      @assert dq_[3] isa AbstractVector{<: ACE.DState}
      @assert length(dq_[3]) == length(cfg)
      dq = dq_[3]  # Vector of DStates
      
      # adj_n = ∑_j dq_j ⋅ ∂B_k / ∂r_j * θ_nk
      # dp ⋅ adj = ∑_n ∑_j dq_j ⋅ ∂B_k / ∂r_j * θ_nk * dp_n 
      # grad[k] = ∑_j dq_j ⋅ ∂B_k / ∂r_j
      grad = ACE.adjoint_EVAL_D1(model, model.evaluator, cfg, dq)

      # gradient w.r.t θ: 
      sdp = SVector(dp...)
      grad_θ = grad .* Ref(sdp)

      # gradient w.r.t. dp    # TODO: remove the |> Vector? 
      grad_dp = sum( model.c[k] * grad[k] for k = 1:length(grad) )  |> Vector 

      return NoTangent(), grad_dp, grad_θ, NoTangent()
   end

   return adj_evaluate(dp, model, cfg), _second_adj
end
