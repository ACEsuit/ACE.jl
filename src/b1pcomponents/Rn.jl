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
function Rn1pBasis(  R::OrthPolyBasis, trans = nothing; 
                     varsym = :rr, nsym = :n, label="R$nsym", )
   spec = [ NamedTuple{(nsym,)}((i,)) for i = 1:length(R) ]
   degrees = collect(0:(length(R)-1))
   R1 = chain(λ("rr -> norm(rr)"), trans, R)
   Rn = B1pComponent(R1, GetVal{varsym}(), spec, degrees, label)
   rl = ACE.inv_transform(r -> Base.invokelatest(trans, r), R.tl)
   rr = ACE.inv_transform(r -> Base.invokelatest(trans, r), R.tr)
   rin, rcut = extrema([rl, rr])
   Rn.meta["rin"] = rin 
   Rn.meta["rcut"] = rcut 
   return Rn 
end
