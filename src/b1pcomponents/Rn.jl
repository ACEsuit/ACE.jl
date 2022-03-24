import ACE.OrthPolys: TransformedPolys, OrthPolyBasis
import ACE.Transforms: GetNorm

@doc raw"""
`Rn1pBasis`

One-particle basis of the form $R_n(r_{ij})$, i.e.,
no dependence on species or on $l$.

This does two things: (1) translates the `TransformedPolys` into a valid
one-particle basis; and (2) treat it as having vectorial input, i.e. value is 
scalar but gradient is vectorial.

The default symbols are `:rr` for the state and `:n` for the index of the 
basis function. 
"""
function Rn1pBasis(R::Union{TransformedPolys, OrthPolyBasis};
                   varsym = :rr, nsym = :n, label="R$nsym", ) 
   spec = [ NamedTuple{(nsym,)}((i,)) for i = 1:length(R) ]
   return B1pComponent(R, GetNorm{varsym}(), spec, label)
end
