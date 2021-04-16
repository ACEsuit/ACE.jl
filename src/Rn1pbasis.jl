
import ACE.OrthPolys: TransformedPolys



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

# ---------------------- Implementation of Rn1pBasis


Base.length(basis::Rn1pBasis) = length(basis.R)

# -> TODO : figure out how to do this well!!!
# Base.rand(basis::Ylm1pBasis) =
#       AtomState(rand(basis.zlist.list), ACE.Random.rand_vec(basis.J))

get_spec(basis::Rn1pBasis) =
      [ (n = n) for n = 1:length(basis) ]

==(P1::Rn1pBasis, P2::Rn1pBasis) =  ACE._allfieldsequal(P1, P2)

write_dict(basis::Rn1pBasis{T}) where {T} = Dict(
      "__id__" => "ACE_Rn1pBasis",
          "R" => write_dict(basis.R) )

read_dict(::Val{:ACE_Rn1pBasis}, D::Dict) = Rn1pBasis(read_dict(D["R"]))

fltype(basis::Rn1pBasis{T}) where T = T

gradtype(::Rn1pBasis{T}) where {T} = SVector{3, T}

symbols(::Rn1pBasis) = [:n]

indexrange(basis::Rn1pBasis) = Dict( :n => 1:length(basis) )

isadmissible(b, basis::Rn1pBasis) = (1 <= b.n <= length(basis))

degree(b, basis::Rn1pBasis) = b.n - 1

get_index(basis::Rn1pBasis, b) = b.n

rand_radial(basis::Rn1pBasis) = rand_radial(basis.R)

# ---------------------------  Evaluation code
#

alloc_B(basis::Rn1pBasis, args...) = alloc_B(basis.R)

alloc_dB(basis::Rn1pBasis, args...) =
      zeros( SVector{3, fltype(basis.R)}, length(basis) )

alloc_temp(basis::Rn1pBasis, args...) = alloc_temp(basis.R)

alloc_temp_d(basis::Rn1pBasis, args...) =
      (
      # alloc_temp_d(basis.R)...,
      dRdr = zeros(fltype(basis.R), length(basis.R)),
      )


evaluate!(B, tmp, basis::Rn1pBasis, X::AbstractState) =
      evaluate!(B, tmp, basis.R, norm(X.rr))

function evaluate_d!(dB, tmpd, basis::Rn1pBasis, X::AbstractState)
   r = norm(X.rr)
   r̂ = X.rr / r
   evaluate_d!(tmpd.dRdr, tmpd, basis.R, r)
   dB[:] .= Ref(r̂) .* tmpd.dRdr
   return dB
end

function evaluate_ed!(B, dB, tmpd, basis::Rn1pBasis, X::AbstractState)
   r = norm(X.rr)
   r̂ = X.rr / r
   evaluate!(B, tmpd, basis.R, norm(X.rr))
   evaluate_d!(tmpd.dRdr, tmpd, basis.R, r)
   dB[:] .= Ref(r̂) .* tmpd.dRdr
   return nothing
end
