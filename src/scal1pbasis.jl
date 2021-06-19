

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
mutable struct Scal1pBasis{VSYM, ISYM, T, TT, TJ} <: OneParticleBasis{T}
   P::TransformedPolys{T, TT, TJ}
end

scal1pbasis(varsym::Symbol, idxsym::Symbol, args...; kwargs...) = 
            Scal1pBasis(varsym, idxsym,  
                  ACE.OrthPolys.transformed_jacobi(args...; kwargs...))

Scal1pBasis(varsym::Symbol, idxsym::Symbol, P::TransformedPolys{T, TT, TJ}
            ) where {T, TT, TJ} = 
      Scal1pBasis{varsym, idxsym, T, TT, TJ}(P)

_varsym(basis::Scal1pBasis{VSYM}) where {VSYM} = VSYM

_idxsym(basis::Scal1pBasis{VSYM, ISYM}) where {VSYM, ISYM} = ISYM

_val(X::AbstractState, basis::Scal1pBasis) = 
   getproperty(X, _varsym(basis))

_val(x::Number, basis::Scal1pBasis) = x

# ---------------------- Implementation of Scal1pBasis


Base.length(basis::Scal1pBasis) = length(basis.P)

get_spec(basis::Scal1pBasis) =
      [  NamedTuple{(_idxsym(basis),)}(n) for n = 1:length(basis) ]

==(P1::Scal1pBasis, P2::Scal1pBasis) =  ACE._allfieldsequal(P1, P2)

write_dict(basis::Scal1pBasis{T}) where {T} = Dict(
      "__id__" => "ACE_Scal1pBasis",
          "P" => write_dict(basis.P) , 
          "varsym" => string(_varsym(basis)), 
          "idxsym" => string(_idxsym(basis)) )

read_dict(::Val{:ACE_Scal1pBasis}, D::Dict) =   
      Scal1pBasis(Symbol(D["varsym"]), Symbol(D["idxsym"]), read_dict(D["P"]))

@noinline valtype(basis::Scal1pBasis, cfg::AbstractConfiguration) = 
      valtype(basis, zero(eltype(cfg)))

@noinline valtype(basis::Scal1pBasis, X::AbstractState) = 
      valtype(basis.P, _val(X, basis))

@noinline gradtype(basis::Scal1pBasis, X::AbstractState) = 
      dstate_type(valtype(basis, X), X)

symbols(basis::Scal1pBasis) = [ _idxsym(basis) ]

indexrange(basis::Scal1pBasis) = NamedTuple{(_idxsym(basis), )}((1:length(basis),))

_getidx(b, basis::Scal1pBasis) = b[_idxsym(basis) ]

isadmissible(b, basis::Scal1pBasis) = (1 <= _getidx(b, basis) <= length(basis))

degree(b, basis::Scal1pBasis) = _getidx(b, basis) - 1

get_index(basis::Scal1pBasis, b) = _getidx(b, basis)

rand_radial(basis::Scal1pBasis) = rand_radial(basis.P)

# ---------------------------  Evaluation code
#

alloc_temp(basis::Scal1pBasis, X::AbstractState) = 
      alloc_temp(basis.P, _val(X, basis))

alloc_temp_d(basis::Scal1pBasis, X::AbstractState) = 
      ( 
         tmpdP = alloc_temp_d(basis.P, _val(X, basis)), 
         dBP = alloc_dB(basis.P, _val(X, basis))
      )



evaluate!(B, tmp, basis::Scal1pBasis, X::AbstractState) =
      evaluate!(B, tmp, basis, _val(X, basis))

evaluate!(B, tmp, basis::Scal1pBasis, x::Number) =
      evaluate!(B, tmp, basis.P, x)

function evaluate_d!(dB, tmpd, basis::Scal1pBasis, X::AbstractState)
   TDX = eltype(dB)
   evaluate_d!(tmpd.dBP, tmpd.tmpdP, basis.P, _val(X, basis))
   for n = 1:length(basis)
      dB[n] = TDX( NamedTuple{(_varsym(basis),)}((tmpd.dBP[n],)) )
   end
   return dB
end

function evaluate_ed!(B, dB, tmpd, basis::Scal1pBasis, X::AbstractState)
   TDX = eltype(dB)
   x = _val(X, basis)
   evaluate!(B, tmpd.tmpdP, basis.P, x)
   evaluate_d!(tmpd.dBP, tmpd.tmpdP, basis.P, x)
   for n = 1:length(basis)
      dB[n] = TDX( NamedTuple{(_varsym(basis),)}((tmpd.dBP[n],)) )
   end
   return B, dB
end


# -------------- AD codes 

import ChainRules: rrule, NO_FIELDS

function _rrule_evaluate(basis::Scal1pBasis, X::AbstractState, 
                         w::AbstractVector{<: Number})
   x = _val(X, basis)
   a = _rrule_evaluate(basis.P, x, real.(w))
   TDX = ACE.dstate_type(a, X)
   return TDX( NamedTuple{(_varsym(basis),)}( (a,) ) )
end

rrule(::typeof(evaluate), basis::Scal1pBasis, X::AbstractState) = 
                  evaluate(basis, X), 
                  w -> (NO_FIELDS, NO_FIELDS, _rrule_evaluate(basis, X, w))

             
                  
function _rrule_evaluate_d(basis::Scal1pBasis, X::AbstractState, 
                           w::AbstractVector)
   x = _val(X, basis)
   w1 = [ _val(w, basis) for w in w ]
   a = _rrule_evaluate_d(basis.P, x, w1)
   TDX = ACE.dstate_type(a, X)
   return TDX( NamedTuple{(_varsym(basis),)}( (a,) ) )
end

function rrule(::typeof(evaluate_d), basis::Scal1pBasis, X::AbstractState)
   x = _val(X, basis)
   dB_ = evaluate_d(basis.P, x)
   TDX = dstate_type(valtype(basis, X), X)
   dB = [ TDX( NamedTuple{(_varsym(basis),)}( (dx,) ) )  for dx in dB_ ]
   return dB, 
          w -> (NO_FIELDS, NO_FIELDS, _rrule_evaluate_d(basis, X, w))
end
