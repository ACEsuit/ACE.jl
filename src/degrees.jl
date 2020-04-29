
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


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

degree(d::SparsePSHDegree, pphi::VecOrTup) =
         d.csp * sum(  d(phi) for phi in pphi ) +
         d.chc * prod( max(d.ahc, d.bhc + d(phi)) for phi in pphi )


"""
`function _get_1p_spec`: Construct the specification for a general 1-particle basis.
"""
function _get_1p_spec(J::ScalarBasis, D::AbstractDegree)
   error("not implemented")
end



@doc raw"""
`function _get_PSH_1p_spec`

Construct the specification for a ``P \otimes Y`` type 1-particle basis.
These must be treated differently because of the requirements that complete
``l``-blocks are represented in the basis.

See also: `_get_1p_spec`.
"""
function _get_PSH_1p_spec(J::ScalarBasis, D::AbstractDegree)
   # find out what the largest degree is that we can allow:
   maxdeg = maximum(D(PSH1pBasisFcn(n, 0, 0, 0)) for n = 1:length(J))

   # generate the `spec::Vector{PSH1pBasisFcn}` using length(J)
   specnl = gensparse(2, maxdeg;
                      tup2b = t -> PSH1pBasisFcn(t[1]+1, t[2], 0, 0),
                      degfun = t -> D(t),
                      ordered = false)
   # add the m-parameters
   return [ PSH1pBasisFcn(b.n, b.l, b.m, 0)
              for b in specnl for m = -b.l:b.l ]
end
