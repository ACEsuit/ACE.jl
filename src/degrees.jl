
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@doc raw"""
`struct PSH1pBasisFunction` : 1-particle basis function specification
for bases of the type ``P \otimes Y`` with `P::ScalarBasis` and `Y::SHBasis`
"""
struct PSH1pBasisFunction
   n::Int
   l::Int
   m::Int
end

PSH1pBasisFunction(t::VecOrTup) = (
      @assert length(t) == 3;
      PSH1pBasisFunction(t[1], t[2], t[3])
   )

Base.show(io::IO, b::PSH1pBasisFunction) = print(io, "nlm[$(b.n),$(b.l),$(b.m)]")



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

degree(d::SparsePSHDegree, phi::PSH1pBasisFunction) = phi.n + d.wL * phi.l

degree(d::SparsePSHDegree, pphi::VecOrTup) =
         d.csp * sum(  d(phi) for phi in pphi ) +
         d.chc * prod( max(d.ahc, d.bhc + d(phi)) for phi in pphi )
