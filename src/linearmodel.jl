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

struct LinearACEModel{TB, TP <: Number, RF} <: ACEModel 
   basis::TB
   c::Vector{TP}
   # TODO: add an evaluator to make the evaluation w.r.t. 
   #       configuration much faster; especially gradients
end 

params(m::LinearACEModel) = copy(m.c)

function set_params!(m::LinearACEModel, c) 
   m.c[:] .= c
   return m 
end

# Todo: -> generalise, maybe introduce `alloc_grad, alloc_params, ...`
grad_params(m::LinearACEModel, X::AbstractConfiguration) = 
      grad_params!(alloc_B(m.basis), m, X)


function grad_params!(g, m::LinearACEModel, X::AbstractConfiguration) 
   evaluate!(g, tmp, m.basis, X) 
   return g 
end

alloc_temp(m::LinearACEModel) = 
   ( tmpbasis = alloc_temp(m.basis), 
     B = alloc_B(m.basis)
   )

# TODO: -> generalise 
evaluate(m::LinearACEModel, X::AbstractConfiguration) = 
      evaluate!(alloc_temp(m), m, X)

function evaluate!(tmp, m::LinearACEModel, X::AbstractConfiguration)  
   evaluate!(tmp.B, m.basis, X)
   return sum(x * y for (x,y) in zip(m.c, tmp.B))
end 


alloc_temp_d(m::LinearModel, X::AbstractConfiguration) = 
      alloc_temp_d(m::LinearModel, length(X))

alloc_temp_d(m::LinearACEModel, N::Integer) = 
      ( tmpdbasis = alloc_temp_d(m.basis), 
        B = alloc_B(m.basis), 
        dB = alloc_dB(m.basis, N)
      )

alloc_grad_config(m::LinearACEModel, X::AbstractConfiguration) = 
      Vector{gradtype(m.basis)}(undef, length(X))

grad_config(m::LinearACEModel, X::AbstractConfiguration) = 
      grad_config!(alloc_grad_config(m, X), alloc_temp_d(m, X), m, X)

function grad_config!(g, tmpd, m::LinearACEModel, X::AbstractConfiguration)
   evaluate_d!(tmpd.B, tmpd.dB, m, X) 
   fill!(g, zero(eltype(g)))
   for ix = 1:length(X), ib = 1:length(m.basis)
      g[ix] += m.c[ib] * dB[ib, ix]
   end
   return g 
end

