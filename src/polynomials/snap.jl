#
#   WORK IN PROGRESS - CURRENTLY NOT INCLUDED
#
#

module SNAP

"""
`SNAPRnlBasis:` define radial basis for hyperspherical harmonics. 
```math
R_{nl}(r) = (-i)^l 2^{l+1/2} l! \cdot 
      \bigg( \frac{(n+1) (n-l)!}{\pi (n+l+1)!} \bigg)^{1/2} \cdot 
      \sin^l\Big( \frac{\pi r}{2 r_0}\Big) \cdot 
      C_{n-l}^{l+1}\Big( \cos\Big(\frac{\pi r}{2 r_0}\Big)\Big).
```
"""
struct SNAPRnlBasis{T, VSYM, LSYM, MSYM}
   maxn::Int 
   maxl::Int 
   spec
   B_pool::VectorPool{Complex{T}}
   dB_pool::VectorPool{T}
end 

_varsym(::SNAPRnlBasis{T, VSYM, LSYM, MSYM}) where {T, VSYM, LSYM, MSYM} = VSYM
_nsym(::SNAPRnlBasis{T, VSYM, LSYM, MSYM}) where {T, VSYM, LSYM, MSYM} = NSYM
_n(b, basis::SNAPRnlBasis) = getproperty(b, _nsym(basis))
_rr(X, Rn::SNAPRnlBasis) = getproperty(X, _varsym(Rn))




function evaluate!(Rnl, basis::SNAPRnlBasis, X::AbstractState)
   rr = _rr(X) 
   r = norm(rr) 
   

end

end