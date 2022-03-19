

import ACE.OrthPolys: OrthPolyBasis, transformed_jacobi

@doc raw"""
`Scal1pBasis`

One-particle basis of the form $P_n(x_i)$ for a general scalar, invariant 
input `x`. This type basically just translates the `TransformedPolys` into a valid
one-particle basis.
"""
function Scal1pBasis(varsym::Symbol, varidx::Union{Integer, Nothing}, idxsym::Symbol, 
                     P, label::String = "P$idxsym", trans=nothing)
   getval = isnothing(varidx) ? ACE.GetVal{varsym}() : ACE.GetVali{varsym, varidx}()
   spec = [ NamedTuple{(idxsym,)}((i,)) for i = 1:length(P) ]
   P1 = trans == nothing ? P : chain(trans, P)
   return B1pComponent(P1, getval, spec, label)
end

          
function scal1pbasis(varsym::Symbol, idxsym::Symbol, 
                     maxdeg::Integer, trans, rcut::Real, rin::Real=0.0; 
                     varidx = nothing, label::String = "P$idxsym", 
                     kwargs...)
   J = transformed_jacobi(maxdeg, trans, rcut, rin; kwargs...)
   @assert J.F[1] == trans 
   @assert J.F[2] isa OrthPolyBasis
   @assert length(J) == 2
   return Scal1pBasis(varsym, varidx, idxsym, J.F[2], label, trans) 
end

