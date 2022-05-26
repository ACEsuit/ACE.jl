
import ACE.Transforms: GetVal

@doc raw"""
`Ylm1pBasis`

One-particle basis component of the form
```math
\phi_{lm}({\bm r}) = Y_l^m(\hat{\br r})
```
Fundamental building block of ACE basis sets of the form
```math
   R_{nl}^{\mu_i, \mu_j}(r_{ij}) Y_l^m(\hat{\bm r})
```
This type basically just translates the `SHBasis` into a valid one-particle
basis.
"""
function Ylm1pBasis(L::Integer; varsym = :rr, lsym = :l, msym = :m, 
                                  label = "Y$lsym$msym")
   SH = SphericalHarmonics.SHBasis(L)
   NTLM = NamedTuple{(lsym, msym)}
   spec = [ NTLM(tuple(SphericalHarmonics.idx2lm(i)...))
            for i = 1:length(SH) ]
   degrees = [ b[lsym] for b in spec ]
   return B1pComponent(SH, GetVal{varsym}(), spec, degrees, label)
end


