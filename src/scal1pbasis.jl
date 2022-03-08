

import ACE: OneParticleBasis, AbstractState
import ACE.OrthPolys: TransformedPolys

import ForwardDiff
import NamedTupleTools
using NamedTupleTools: namedtuple


@doc raw"""
`Scal1pBasis`

One-particle basis of the form $P_n(x_i)$ for a general scalar, invariant 
input `x`. This type basically just translates the `TransformedPolys` into a valid
one-particle basis.
"""
function Scal1pBasis(varsym::Symbol, varidx::Union{Integer, Nothing}, idxsym::Symbol, 
                     P::TransformedPolys, label::String = "")
   getval = isnothing(varidx) ? ACE.GetVal{varsym}() : ACE.getVali{varsym, varidx}()
   B = XScal1pBasis(P, 
                    NamedTuple{(idxsym,)}( (1:length(P),) ),
                    getval, 
                    label ) 
   init1pspec!(B, SimpleSparseBasis(1, length(P)))
   fill_diag_coeffs!(B)
   return B    
end


scal1pbasis(varsym::Symbol, idxsym::Symbol, args...; varidx = nothing, label = "", kwargs...) = 
            Scal1pBasis(varsym, varidx, idxsym,  
                  ACE.OrthPolys.transformed_jacobi(args...; kwargs...), 
                  label)
