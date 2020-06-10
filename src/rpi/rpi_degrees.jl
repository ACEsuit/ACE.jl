
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

export SparsePSHDegree

@doc raw"""
`struct PSH1pBasisFcn` : 1-particle basis function specification
for bases of the type ``P \otimes Y`` with `P::ScalarBasis` and `Y::SHBasis`
"""
struct PSH1pBasisFcn <: OnepBasisFcn
   n::Int
   l::Int
   m::Int
   z::AtomicNumber
end

function PSH1pBasisFcn(t::VecOrTup)
   if length(t) == 3
      return PSH1pBasisFcn(t[1], t[2], t[3], 0)
   elseif length(t) == 4
      return PSH1pBasisFcn(t...)
   end
   error("`PSH1pBasisFcn(t::VecOrTup)` : `t` must have length 3 or 4")
end

Base.show(io::IO, b::PSH1pBasisFcn) = print(io, "znlm[$(b.z.z)|$(b.n),$(b.l),$(b.m)]")

write_dict(b::PSH1pBasisFcn) =
   Dict("__id__" => "SHIPs_PSH1pBasisFcn",
        "nlmz" => [ b.n, b.l, b.m, Int(b.z) ] )

read_dict(::Val{:SHIPs_PSH1pBasisFcn}, D::Dict) =
   PSH1pBasisFcn(D["nlmz"]...)

scaling(b::PSH1pBasisFcn, p) = b.n^p + b.l^p

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
@with_kw struct SparsePSHDegree <: AbstractDegree
   wL::Float64   = 1.5
   csp::Float64  = 1.0
   chc::Float64  = 0.0
   ahc::Float64  = 0.0
   bhc::Float64  = 0.0
end

degree(d::SparsePSHDegree, phi::PSH1pBasisFcn) = phi.n + d.wL * phi.l

function degree(d::SparsePSHDegree, pphi::VecOrTup)
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

Note: at the moment, the z0-dependence doesn't work, this needs some reworking
of internals. The third argument should therefore be ignored for now!

This is very awkward of course, so there is an alternative constructor,
```
SparsePSHDegreeM(Dn::Dict, Dl::Dict)
```
which will construct the functions `wN_fun, wL_fun` by checking for information
in the two dictionaries in the following order of precedence:
- Look for a key `(N, zi, z0)`
- Look for a key `N`
- Look for a key `"default"`
If none exist, an error is thrown.
"""
@with_kw struct SparsePSHDegreeM <: AbstractDegree
   wNfun
   wLfun
end

degree(d::SparsePSHDegreeM, phi::PSH1pBasisFcn) =
      (    d.wNfun(1, phi.z, phi.z) * phi.n
         + d.wLfun(1, phi.z, phi.z) * phi.l )

function degree(d::SparsePSHDegreeM, pphi::VecOrTup)
   if length(pphi) == 0
      return 0
   end
   N = length(pphi)
   return sum( (  d.wNfun(N, phi.z, phi.z) * phi.n
                + d.wLfun(N, phi.z, phi.z) * phi.l)    for phi in pphi )
end

function _readfromdict(D, args)
   if haskey(D, args)
      return D[args]
   elseif haskey(D, args[1])
      return D[args[1]]
   elseif haskey(D, "default")
      return D["default"]
   end
   error("SparsePSHDegreeM: no valid key found for argument $(args)")
end

SparsePSHDegreeM(DN::Dict, DL::Dict) =
      SparsePSHDegreeM( (N, zi, z0) -> _readfromdict(DN, (N, zi, z0)),
                        (N, zi, z0) -> _readfromdict(DL, (N, zi, z0)) )
