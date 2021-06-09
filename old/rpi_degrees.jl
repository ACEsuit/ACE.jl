
# --------------------------------------------------------------------------
# ACE.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



export SparsePSHDegree

using Parameters: @with_kw

abstract type AbstractPSHDegree <: AbstractDegree end

get_maxn(d::AbstractPSHDegree, maxdeg, species::Symbol) =
   get_maxn(d, maxdeg, (species,))

function get_maxn(d::AbstractPSHDegree, maxdeg, species)
   maxn = 0
   for s1 in species, s2 in species
      n = 1
      z  = AtomicNumber(s1)
      z0 = AtomicNumber(s2)
      while degree(d, RnYlmBasisFcn(n, 0, 0)) < maxdeg
         n += 1
      end
      maxn = max(maxn, n)
   end
   return maxn
end


@doc raw"""
`SparsePSHDegree` : A general sparse-grid type degree definition for
``Pr \otimes Y`` type basis functions
```math
{\rm deg}(\{n_i\}, \{l_i\})
= c_{\rm sp} \sum_i (n_i + w_{\rm L} l_i)
  + c_{\rm hc} \prod_i \max(a_{\rm hc}, b_{\rm hc} + n_i + w_{\rm L} * l_i)
```

### Constructor
```julia
SparsePSHDegree(wL = 1.5, csp = 1.0, chc = 0.0, ahc = 0.0, bhc = 0.0)
```
"""
@with_kw struct SparsePSHDegree <: AbstractPSHDegree
   wL::Float64   = 1.5
   csp::Float64  = 1.0
   chc::Float64  = 0.0
   ahc::Float64  = 0.0
   bhc::Float64  = 0.0
end


degree(d::SparsePSHDegree, phi::RnYlmBasisFcn, z0=nothing) =
      phi.n + d.wL * phi.l

function degree(d::SparsePSHDegree, pphi::VecOrTup, z0=nothing)
   if length(pphi) == 0
      return 0
   else
      return (
         d.csp * sum(  d(phi) for phi in pphi ) +
         d.chc * prod( max(d.ahc, d.bhc + d(phi)) for phi in pphi )
      )
   end
end



@doc raw"""
`SparsePSHDegreeM` : A general sparse-grid type degree definition for
``P_r \otimes Y`` type basis functions, which gives more freedom to adjust
the weights across species and correlation-orders. For simplicity, this doesn't
admit hyperbolic-cross type constructions but only the classical sparse grid.
```math
{\rm deg}(\{n_i\}_{i=1}^N, \{l_i\}_{i=1}^N)
= \sum_{i=1}^N (w^{\rm n}_i n_i + w^{\rm l}_i l_i)
```
where ``w^{\rm n}_i, w^{\rm l}_i`` may now depend on ``z_i, z_0, N``.

### Constructor

```julia
SparsePSHDegreeM(wn_fun, wl_fun)
```
where `wn_fun, wlfun` are functions that must take the arguments `(N, zi, z0)`.

### A More Practical Constructor

The functional constructor is very awkward to use, so there is an alternative
constructor, which constructs the functions wn_fun, wl_fun from a few
dictionaries.
```
SparsePSHDegreeM(Dn::Dict, Dl::Dict, Dd::Dict)
```
which will construct the functions `wn_fun, wl_fun` as follows:
```
wn_fun(N, zi, z0) = Dn[(N, zi, z0)] / Dd[(N, z0)]
wl_fun(N, zi, z0) = Dl[(N, zi, z0)] / Dd[(N, z0)]
```
If the key `(N, zi, z0)` does not exist then it will instead look for
- a key `(zi, z0)`
- or a key `"default"`
If none exist, an error is thrown. Similarly if the key `(N, z0)` does not
exist, then it will look for
- a key `N`
- a key `z0`
- a key "default".
If none of these exist, then it will throw an error.
"""
struct SparsePSHDegreeM <: AbstractPSHDegree
   wNfun
   wLfun
end


degree(d::SparsePSHDegreeM, phi::RnYlmBasisFcn, z0::AtomicNumber) =
      (    d.wNfun(1, phi.z, z0) * phi.n
         + d.wLfun(1, phi.z, z0) * phi.l )

# function degree(d::SparsePSHDegreeM, b::PIBasisFcn)
#    pphi = b.oneps
#    z0 = b.z0
#    N = length(pphi)
#    if N == 0; return 0; end
#    return sum( (  d.wNfun(N, phi.z, z0) * phi.n
#                 + d.wLfun(N, phi.z, z0) * phi.l)    for phi in pphi )
# end

function _finddegree(D::Dict, N, z0)
   if haskey(D, (N, z0))
      return D[(N, z0)]
   elseif haskey(D, N)
      return D[N]
   elseif haskey(D, z0)
      return D[z0]
   elseif haskey(D, "default")
      return D["default"]
   end
   error("can't find a valid degree")
end

function _findweight(D::Dict, N, zi, z0)
   if haskey(D, (N, zi, z0))
      return D[(N, zi, z0)]
   elseif haskey(D, (zi, z0))
      return D[(zi, z0)]
   elseif haskey(D, "default")
      return D["default"]
   end
   error("SparsePSHDegreeM: no valid key found for argument $(args)")
end

SparsePSHDegreeM(DN::Dict, DL::Dict, DD::Dict) = SparsePSHDegreeM(
         (N, zi, z0) -> _findweight(DN, N, zi, z0) / _finddegree(DD, N, z0),
         (N, zi, z0) -> _findweight(DL, N, zi, z0) / _finddegree(DD, N, z0) )
