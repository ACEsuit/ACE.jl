
import ACE.OrthPolys: TransformedPolys

# TODO: it feels like there should be a generic wrapper implementation which 
#       unifies Rn, Ylm and Pk and then just needs a tiny bit of wrapping...

@doc raw"""
`Rn1pBasis <: OneParticleBasis`

One-particle basis of the form $R_n(r_{ij})$, i.e.,
no dependence on species or on $l$.

This does two things: (1) translates the `TransformedPolys` into a valid
one-particle basis; and (2) treat it as having vectorial input, i.e. value is 
scalar but gradient is vectorial.

The default symbols are `:rr` for the state and `:n` for the index of the 
basis function. 
"""
function Rn1pBasis(R::TransformedPolys; label="Rn", varsym = :rr, nsym = :n,
                                        Bsel = SimpleSparseBasis(1, length(R)) )
   Rn = xscal1pbasis(R, 
                     NamedTuple{(nsym,)}( (1:length(R),) ), 
                     ACE.GetNorm{varsym}(); 
                     label = label)
   init1pspec!(Rn, Bsel)
   fill_diag_coeffs!(Rn)
   return Rn 
end                      
