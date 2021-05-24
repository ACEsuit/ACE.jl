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
end 

struct NaiveEvaluator end 

function LinearACEModel(basis::SymmetricBasis, c = zeros(length(basis)); 
               evaluator = :standard) 
   if evaluator == :naive 
      ev = NaiveEvaluator()
   elseif evaluator == :standard 
      ev = PIEvaluator(basis, c) 
   elseif evaluator == :recursive 
      error("Recursive evaluator not yet implemented")
   else 
      error("unknown evaluator")
   end
   return LinearACEModel(basis, c, ev)
end

params(m::LinearACEModel) = copy(m.c)

function set_params!(m::LinearACEModel, c) 
   m.c[:] .= c
   set_params!(m.evaluator, m.basis, c)
   return m 
end

set_params!(::NaiveEvaluator, args...) = nothing 

# ------------------- dispatching on the evaluators 

alloc_temp(m::LinearACEModel) = alloc_temp(m.evaluator, m)

evaluate(m::LinearACEModel, X::AbstractConfiguration) = 
      evaluate!(alloc_temp(m), m, X)

evaluate!(tmp, m::LinearACEModel, X::AbstractConfiguration) = 
      evaluate!(tmp, m::LinearACEModel, m.evaluator, X::AbstractConfiguration)

alloc_temp_d(m::LinearACEModel, X::AbstractConfiguration) = 
      alloc_temp_d(m, length(X))

alloc_temp_d(m::LinearACEModel, N::Integer) = 
      alloc_temp_d(m.evaluator, N, m)

# this one seems generic and doesn't need to be dispatched?
alloc_grad_config(m::LinearACEModel, X::AbstractConfiguration) = 
      Vector{gradtype(m.basis)}(undef, length(X))

grad_config(m::LinearACEModel, X::AbstractConfiguration) = 
      grad_config!(alloc_grad_config(m, X), alloc_temp_d(m, X), m, X)

grad_config!(g, tmpd, m::LinearACEModel, X::AbstractConfiguration) = 
      grad_config!(g, tmpd, m, m.evaluator, X) 

grad_params(m::LinearACEModel, X::AbstractConfiguration) = 
      grad_params!(alloc_B(m.basis), alloc_temp(m.basis), m, X)

function grad_params!(g, tmp, m::LinearACEModel, X::AbstractConfiguration) 
   evaluate!(g, tmp, m.basis, X) 
   return g 
end

function grad_params_config(m::LinearACEModel, X::AbstractConfiguration) 
   tmpd = alloc_temp_d(m.basis, length(X))
   dB = alloc_dB(m.basis, length(X))
   return grad_params_config!(dB, tmpd, m, X)
end


grad_params_config!(dB, tmpd, m::LinearACEModel, X::AbstractConfiguration)  = 
      evaluate_d!(dB, tmpd, m.basis, X) 

# ------------------- implementation of naive evaluator 

alloc_temp(::NaiveEvaluator, m::LinearACEModel) = 
   ( tmpbasis = alloc_temp(m.basis), 
     B = alloc_B(m.basis)
   )

function evaluate!(tmp, m::LinearACEModel, ::NaiveEvaluator, 
                    X::AbstractConfiguration)  
   evaluate!(tmp.B, tmp.tmpbasis, m.basis, X)
   return sum(prod, zip(m.c, tmp.B))
end 

alloc_temp_d(::NaiveEvaluator, N::Integer, m::LinearACEModel) = 
      ( tmpdbasis = alloc_temp_d(m.basis, N), 
        B = alloc_B(m.basis), 
        dB = alloc_dB(m.basis, N)
      )

function grad_config!(g, tmpd, m::LinearACEModel, ::NaiveEvaluator, 
                     X::AbstractConfiguration)
   evaluate_d!(tmpd.dB, tmpd.tmpdbasis, m.basis, X) 
   fill!(g, zero(eltype(g)))
   for ix = 1:length(X), ib = 1:length(m.basis)
      g[ix] += m.c[ib] * tmpd.dB[ib, ix]
   end
   return g 
end

