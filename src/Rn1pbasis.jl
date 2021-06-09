
import ACE.OrthPolys: TransformedPolys



@doc raw"""
`struct Rn1pBasis <: OneParticleBasis`

One-particle basis of the form $R_n(r_{ij})$, i.e.,
no dependence on species or on $l$.

This does two things: (1) translates the `TransformedPolys` into a valid
one-particle basis; and (2) treat it as having vectorial input, i.e. value is 
scalar but gradient is vectorial.

The default symbols are `:rr` for the state and `:n` for the index of the 
basis function. 
"""
mutable struct Rn1pBasis{T, TT, TJ, VSYM, NSYM} <: OneParticleBasis{T}
   R::TransformedPolys{T, TT, TJ}
end


Rn1pBasis(R::TransformedPolys{T, TT, TJ}; varsym = :rr, nsym = :n
            ) where {T, TT, TJ} = 
      Rn1pBasis{T, TT, TJ, varsym, nsym}(R)

# ---------------------- Implementation of Rn1pBasis

Base.length(basis::Rn1pBasis) = length(basis.R)

_varsym(::Rn1pBasis{T, TT, TJ, VSYM, NSYM}) where {T, TT, TJ, VSYM, NSYM} = VSYM
_nsym(::Rn1pBasis{T, TT, TJ, VSYM, NSYM}) where {T, TT, TJ, VSYM, NSYM} = NSYM
_n(b, basis::Rn1pBasis) = getproperty(b, _nsym(basis))
_rr(X, Rn::Rn1pBasis) = getproperty(X, _varsym(Rn))

# -> TODO : figure out how to do this well!!!
# Base.rand(basis::Ylm1pBasis) =
#       AtomState(rand(basis.zlist.list), ACE.Random.rand_vec(basis.J))

function get_spec(basis::Rn1pBasis) 
   N = _nsym(basis)
   return [  NamedTuple{(N,)}((n,)) for n = 1:length(basis) ]
end

==(P1::Rn1pBasis, P2::Rn1pBasis) =  ACE._allfieldsequal(P1, P2)

write_dict(basis::Rn1pBasis{T}) where {T} = Dict(
      "__id__" => "ACE_Rn1pBasis",
          "R" => write_dict(basis.R), 
          "varsym" => _varsym(basis), 
          "nsym" => _nsym(basis) )

read_dict(::Val{:ACE_Rn1pBasis}, D::Dict) = 
            Rn1pBasis(read_dict(D["R"]), 
                      varsym = Symbol(D["varsym"]), 
                      nsym = Symbol(D["nsym"]))

fltype(basis::Rn1pBasis{T}) where T = T

gradtype(B::Rn1pBasis, X::AbstractState) = 
            dstate_type(fltype(B), X)

symbols(Rn::Rn1pBasis) = [ _nsym(Rn) ]

indexrange(Rn::Rn1pBasis) = NamedTuple{(_nsym(Rn),)}((1:length(Rn),))

isadmissible(b, basis::Rn1pBasis) = (1 <= _n(b, basis) <= length(basis))

degree(b, Rn::Rn1pBasis) = _n(b, Rn) - 1

degree(b, Rn::Rn1pBasis, weight::Dict) = weight[_nsym(Rn)] * degree(b, Rn)

get_index(basis::Rn1pBasis, b) = _n(b, basis)

rand_radial(basis::Rn1pBasis) = rand_radial(basis.R)

# ---------------------------  Evaluation code
#

alloc_B(basis::Rn1pBasis, args...) = alloc_B(basis.R)

alloc_dB(basis::Rn1pBasis, X::AbstractState) =
      zeros( gradtype(basis, X), length(basis) )

alloc_temp(basis::Rn1pBasis, args...) = alloc_temp(basis.R)

alloc_temp_d(basis::Rn1pBasis, args...) =
      (
      # alloc_temp_d(basis.R)...,
      dRdr = zeros(fltype(basis.R), length(basis.R)),
      )


evaluate!(B, tmp, basis::Rn1pBasis, X::AbstractState) =
      evaluate!(B, tmp, basis.R, norm(_rr(X, basis)))

function evaluate_d!(dB, tmpd, basis::Rn1pBasis, X::AbstractState)
   TDX = eltype(dB)
   RR = _varsym(basis)
   rr = _rr(X, basis)
   r = norm(rr)
   r̂ = rr / r
   evaluate_d!(tmpd.dRdr, tmpd, basis.R, r)
   for n = 1:length(basis)
      dB[n] = TDX( NamedTuple{(RR,)}( (tmpd.dRdr[n] * r̂,) ) )
   end
   return dB
end

function evaluate_ed!(B, dB, tmpd, basis::Rn1pBasis, X::AbstractState)
   evaluate!(B, tmpd, basis, X)
   evaluate_d!(dB, tmpd, basis, X)
   return nothing
end
