

import ACE.Transforms: GetVal

@doc raw"""
`Trig1pBasis`

One-particle basis component of the form
```math
\phi_{l}(\theta) = \exp(i l \theta)
```
Fundamental building block of ACE basis sets of the form
```math
\begin{aligned}
   \phi_{nl}({\bm r}) &= R_{n}(r_{ij}) \exp(i l \theta) \\ 
   \phi_{nlm}({\bm r}) &= R_{n}(r_{ij}) \exp(i l \theta) T_{m}(z) \\ 
```
However, for various reasons the implementation takes as input not an angle 
``\theta`` but a 2-dimensional vector ``{\bm r}`` taken to be synonymous with 
```math 
   {\bm r} = r e^{i \theta}.
```
"""
function Trig1pBasis(L::Integer; varsym = :rr, lsym = :l, label = "E$lsym")
   spec = [ (l = l,) for l = -L:L ]
   degrees = ans.(spec)
   return B1pComponent(Trig(L), GetVal{varsym}(), spec, degrees, label)
end


# -------- prototype implementation 
#          a more sophisticated implementation could go into the polynomials 

using ForwardDiff

struct Trig
   L::Int 
end

function cossin(rr)
   r = norm(rr)
   return SVector(rr[1]/r, rr[2]/r)
end

function cossin_ed(rr)
   r = norm(rr)
   c = rr[1]/r 
   s = rr[2]/r
   return SVector(c, s), SVector(-s/r, c/r)
end


_theta(rr) = atan(rr[2], rr[1])

function _theta_ed(rr)
   rsq = rr[1]^2+rr[2]^2
   return _theta(rr), SVector(-rr[2]/rsq, rr[1]/rsq)
end


function evaluate(E::Trig, rr) 
   θ = _theta(rr)
   return [ exp(im * θ * l) for l = -E.L:E.L ]
end

function ACE.evaluate_d(E::Trig, rr)
   θ, dθ = _theta_ed(rr)
   return [ (im * l * exp(im * θ * l)) * dθ  for l = -E.L:E.L ]
end

evaluate_ed(E::Trig, rr) = evaluate(E, rr), evaluate_d(E, rr)
