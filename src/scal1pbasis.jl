

import ACE: OneParticleBasis, AbstractState
import ACE.OrthPolys: TransformedPolys

import NamedTupleTools
using NamedTupleTools: namedtuple


@doc raw"""
`struct Scal1pBasis <: OneParticleBasis`

One-particle basis of the form $P_n(x_i)$ for a general scalar, invariant 
input `x`. This type basically just translates the `TransformedPolys` into a valid
one-particle basis.
"""
mutable struct Scal1pBasis{T, TT, TJ} <: OneParticleBasis{T}
   P::TransformedPolys{T, TT, TJ}
   varsym::Symbol
   idxsym::Symbol
end

scal1pbasis(varsym::Symbol, idxsym::Symbol, args...; kwargs...) = 
            Scal1pBasis( ACE.OrthPolys.transformed_jacobi(args...; kwargs...), 
                         varsym, idxsym )

# ---------------------- Implementation of Scal1pBasis


Base.length(basis::Scal1pBasis) = length(basis.P)

get_spec(basis::Scal1pBasis) =
      [  NamedTuple{(basis.idxsym,)}(n) for n = 1:length(basis) ]

==(P1::Scal1pBasis, P2::Scal1pBasis) =  ACE._allfieldsequal(P1, P2)

write_dict(basis::Scal1pBasis{T}) where {T} = Dict(
      "__id__" => "ACE_Scal1pBasis",
          "P" => write_dict(basis.P) , 
          "varsym" => string(basis.varsym), 
          "idxsym" => string(basis.idxsym) )

read_dict(::Val{:ACE_Scal1pBasis}, D::Dict) =   
      Scal1pBasis(read_dict(D["P"]), Symbol(D["varsym"]), Symbol(D["idxsym"]))

fltype(basis::Scal1pBasis{T}) where T = T

gradtype(basis::Scal1pBasis, X::AbstractState) = 
      dstate_type(fltype(basis), X)

symbols(basis::Scal1pBasis) = [basis.idxsym]

indexrange(basis::Scal1pBasis) = NamedTuple{(basis.idxsym,)}((1:length(basis),))

_getidx(b, basis::Scal1pBasis) = b[basis.idxsym]

isadmissible(b, basis::Scal1pBasis) = (1 <= _getidx(b, basis) <= length(basis))

degree(b, basis::Scal1pBasis) = _getidx(b, basis) - 1

get_index(basis::Scal1pBasis, b) = _getidx(b, basis)

rand_radial(basis::Scal1pBasis) = rand_radial(basis.P)

# ---------------------------  Evaluation code
#

alloc_B(basis::Scal1pBasis, args...) = alloc_B(basis.P)

alloc_dB(basis::Scal1pBasis, X::AbstractState) = 
      zeros(gradtype(basis, X), length(basis))

alloc_temp(basis::Scal1pBasis, args...) = alloc_temp(basis.P)

alloc_temp_d(basis::Scal1pBasis, args...) = 
      ( 
         tmpdP = alloc_temp_d(basis.P), 
         dBP = alloc_dB(basis.P)
      )


_getval(X::AbstractState, basis::Scal1pBasis) = getproperty(X, basis.varsym)
_getval(x::Number, basis::Scal1pBasis) = x

evaluate!(B, tmp, basis::Scal1pBasis, X::AbstractState) =
      evaluate!(B, tmp, basis, _getval(X, basis))

evaluate!(B, tmp, basis::Scal1pBasis, x::Number) =
      evaluate!(B, tmp, basis.P, x)

function evaluate_d!(dB, tmpd, basis::Scal1pBasis, X::TX) where {TX <: AbstractState}
   evaluate_d!(tmpd.dBP, tmpd.tmpdP, basis.P, _getval(X, basis))
   for n = 1:length(basis)
      dB[n] = TX( NamedTuple{(basis.varsym,)}((tmpd.dBP[n],)) )
   end
   return dB
end

function evaluate_ed!(B, dB, tmpd, basis::Scal1pBasis, X::AbstractState)
   TDX = eltype(dB)
   x = _getval(X, basis)
   evaluate!(B, tmpd.tmpdP, basis.P, x)
   evaluate_d!(tmpd.dBP, tmpd.tmpdP, basis.P, x)
   for n = 1:length(basis)
      dB[n] = TDX( NamedTuple{(basis.varsym,)}((tmpd.dBP[n],)) )
   end
   return B, dB
end
