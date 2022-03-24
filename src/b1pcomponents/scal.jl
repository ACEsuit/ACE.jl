

import ACE.OrthPolys: TransformedPolys

@doc raw"""
`Scal1pBasis`

One-particle basis of the form $P_n(x_i)$ for a general scalar, invariant 
input `x`. This type basically just translates the `TransformedPolys` into a valid
one-particle basis.
"""
function Scal1pBasis(varsym::Symbol, varidx::Union{Integer, Nothing}, idxsym::Symbol, 
                       P::TransformedPolys, label::String = "P$idxsym")
   getval = isnothing(varidx) ? ACE.GetVal{varsym}() : ACE.getVali{varsym, varidx}()
   spec = [ NamedTuple{(idxsym,)}((i,)) for i = 1:length(P) ]
   return B1pComponent(P, getval, spec, label)
end

          
scal1pbasis(varsym::Symbol, idxsym::Symbol, args...; 
            varidx = nothing, label::String = "P$idxsym", kwargs...) = 
      Scal1pBasis(varsym, varidx, idxsym,  
                  ACE.OrthPolys.transformed_jacobi(args...; kwargs...), 
                  label)


