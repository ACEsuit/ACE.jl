
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

import ACE.OrthPolys: TransformedPolys

@doc raw"""
`struct Rn1pBasisFcn` : 1-particle basis function specification
for the `Rn1pBasis`.
"""
struct Rn1pBasisFcn <: OnepBasisFcn
   n::Int
end

@doc raw"""
`struct Rn1pBasis <: OneParticleBasis`

One-particle basis of the form $R_n(r_{ij})$, i.e.,
no dependence on species or on $l$.

This type basically just translates the `TransformedPolys` into a valid
one-particle basis.
"""
mutable struct Rn1pBasis{T, TT, TJ} <: OneParticleBasis{T}
   R::TransformedPolys{T, TT, TJ}
end

# TODO: allow for the possibility that the symbol where the
#       position is stored is not `rr` but something else!
#
# TODO: Should we drop this type altogether and just
#       rewrite TransformedPolys to become a 1pbasis?

# ---------------------- Implementation of Rn1pBasisFcn

Rn1pBasisFcn(t::VecOrTup) = Rn1pBasisFcn(t...)

Base.show(io::IO, b::Rn1pBasisFcn) = print(io, "n[$(b.n)]")

write_dict(b::Rn1pBasisFcn) =
   Dict("__id__" => "ACE_Rn1pBasisFcn",
        "n" => b.n )

read_dict(::Val{:ACE_Rn1pBasisFcn}, D::Dict) = Rn1pBasisFcn(D["n"])

scaling(b::Rn1pBasisFcn, p) = b.n^p

degree(b::Rn1pBasisFcn, p) = b.n

# ---------------------- Implementation of Ylm1pBasis

# use default constructor...

cutoff(basis::Rn1pBasis) = cutoff(basis.R)

Base.length(basis::Rn1pBasis) = length(basis.R)

# -> TODO : figure out how to do this well!!!
# Base.rand(basis::Ylm1pBasis) =
#       AtomState(rand(basis.zlist.list), ACE.Random.rand_vec(basis.J))

get_basis_spec(basis::Rn1pBasis) =
      [ Rn1pBasisFcn(n) for n = 1:length(basis) ]

==(P1::Rn1pBasis, P2::Rn1pBasis) =  ACE._allfieldsequal(P1, P2)

write_dict(basis::Rn1pBasis{T}) where {T} = Dict(
      "__id__" => "ACE_Rn1pBasis",
          "R" => write_dict(basis.R) )

read_dict(::Val{:ACE_Rn1pBasis}, D::Dict) = Rn1pBasis(read_dict(D["R"]))

fltype(basis::Rn1pBasis{T}) where T = T


# ---------------------------  Evaluation code
#

alloc_B(basis::Rn1pBasis) = alloc_B(basis.R)

alloc_temp(basis::Rn1pBasis) = alloc_temp(basis.R)

evaluate!(B, tmp, basis::Rn1pBasis, X::AbstractState, X0::AbstractState) =
      evaluate!(B, tmp, basis.R, norm(X.rr))
