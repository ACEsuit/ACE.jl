
import ACE.OrthPolys: TransformedPolys

# TODO: it feels like there should be a generic wrapper implementation which 
#       unifies Rn, Ylm and Pk and then just needs a tiny bit of wrapping...

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
mutable struct Rn1pBasis{T, TT, TJ, VSYM, NSYM, TDX} <: OneParticleBasis{T}
   R::TransformedPolys{T, TT, TJ}
   B_pool::VectorPool{T}
   dB_pool::VectorPool{TDX}
   label::String
end

Rn1pBasis(R::TransformedPolys{T, TT, TJ}; label="Rn", varsym = :rr, nsym = :n
            ) where {T, TT, TJ} = 
      Rn1pBasis{T, TT, TJ, varsym, nsym}(R, label)

function Rn1pBasis{T, TT, TJ, VSYM, NSYM}(R::TransformedPolys{T, TT, TJ}, 
                                          label::String
                                         ) where {T, TT, TJ, VSYM, NSYM} 
   TDX = DState{NamedTuple{(VSYM,), Tuple{SVector{3, T}}}}
   return Rn1pBasis{T, TT, TJ, VSYM, NSYM, TDX}(
               R, VectorPool{T}(), VectorPool{TDX}(), label)
end



# # -------- temporary hack for 1.6, should not be needed from 1.7 onwards 

# function acquire_B!(basis::Rn1pBasis, args...) 
#    VT = valtype(basis, args...)
#    return acquire!(basis.B_pool, length(basis), VT)
# end

# function release_B!(basis::Rn1pBasis, B)
#    return release!(basis.B_pool, B)
# end

# ---------------------- Implementation of Rn1pBasis

Base.length(basis::Rn1pBasis) = length(basis.R)

_varsym(::Rn1pBasis{T, TT, TJ, VSYM, NSYM}) where {T, TT, TJ, VSYM, NSYM} = VSYM
_nsym(::Rn1pBasis{T, TT, TJ, VSYM, NSYM}) where {T, TT, TJ, VSYM, NSYM} = NSYM
_n(b, basis::Rn1pBasis) = getproperty(b, _nsym(basis))
_rr(X, Rn::Rn1pBasis) = getproperty(X, _varsym(Rn))

# -> TODO : figure out how to do this well!!!
# Base.rand(basis::Ylm1pBasis) =
#       AtomState(rand(basis.zlist.list), ACE.Random.rand_vec(basis.J))


function Base.show(io::IO, basis::Rn1pBasis)
   print(io, "Rn1pBasis{$(_varsym(basis)), $(_nsym(basis))}(") 
   print(io, basis.R.J) 
   print(io, ", ")
   print(io, basis.R.trans)
   print(io, ", \"$(basis.label)\")")
end


get_spec(basis::Rn1pBasis, n::Integer) = NamedTuple{(_nsym(basis),)}((n,))

get_spec(basis::Rn1pBasis) = get_spec.(Ref(basis), 1:length(basis))

==(P1::Rn1pBasis, P2::Rn1pBasis) = 
   ( (P1.R == P2.R) && (typeof(P1) == typeof(P2)) )

write_dict(basis::Rn1pBasis{T}) where {T} = Dict(
      "__id__" => "ACE_Rn1pBasis",
          "R" => write_dict(basis.R), 
          "varsym" => _varsym(basis), 
          "nsym" => _nsym(basis), 
          "label" => basis.label )

read_dict(::Val{:ACE_Rn1pBasis}, D::Dict) = 
            Rn1pBasis(read_dict(D["R"]); 
                      varsym = Symbol(D["varsym"]), 
                      nsym = Symbol(D["nsym"]), 
                      label = D["label"] )

# TODO: this seems really poor; should the valtype use a type promotion here 
# what if the input is a Dual???
valtype(basis::Rn1pBasis{T}, args...) where {T} = T

# TODO: this should be a generic fallback I think 
gradtype(B::Rn1pBasis, X::AbstractState) = dstate_type(valtype(B, X), X)

symbols(Rn::Rn1pBasis) = [ _nsym(Rn) ]

argsyms(Rn::Rn1pBasis) = ( _varsym(Rn), )

indexrange(Rn::Rn1pBasis) = NamedTuple{(_nsym(Rn),)}((1:length(Rn),))

isadmissible(b, basis::Rn1pBasis) = (1 <= _n(b, basis) <= length(basis))

degree(b, Rn::Rn1pBasis) = _n(b, Rn) - 1

degree(b, Rn::Rn1pBasis, weight::Dict) = haskey(weight,_nsym(Rn)) ? weight[_nsym(Rn)] * degree(b, Rn) : degree(b, Rn)

get_index(basis::Rn1pBasis, b) = _n(b, basis)

rand_radial(basis::Rn1pBasis) = rand_radial(basis.R)

# ---------------------------  Evaluation code
#

evaluate!(B, basis::Rn1pBasis, X::AbstractState) =
      evaluate!(B, basis.R, norm(_rr(X, basis)))

function evaluate(basis::Rn1pBasis, X::AbstractState)
   rr = _rr(X, basis)
   return evaluate(basis.R, norm(rr))
end


function evaluate_d!(dB, basis::Rn1pBasis, X::AbstractState)
   TDX = eltype(dB)
   RR = _varsym(basis)
   rr = _rr(X, basis)
   r = norm(rr)
   r̂ = rr / r
   dRdr = acquire_dB!(basis.R, r)
   evaluate_d!(dRdr, basis.R, r)
   for n = 1:length(basis)
      dB[n] = TDX( NamedTuple{(RR,)}( (dRdr[n] * r̂,) ) )
   end
   release_dB!(basis.R, dRdr)
   return dB
end

function evaluate_ed!(B, dB, basis::Rn1pBasis, X::AbstractState)
   evaluate!(B, basis, X)
   evaluate_d!(dB, basis, X)
   return B, dB 
end 

# ----------------- AD ... experimental

import ChainRules: rrule, NoTangent

function _rrule_evaluate(basis::Rn1pBasis, X::AbstractState, 
                         w::AbstractVector{<: Number})
   rr = _rr(X, basis)
   r = norm(rr)
   a = _rrule_evaluate(basis.R, r, real.(w))
   TDX = ACE.dstate_type(a/r, X)
   return TDX( NamedTuple{(_varsym(basis),)}( (a * rr / r,)) )
end

rrule(::typeof(evaluate), basis::Rn1pBasis, X::AbstractState) = 
                  evaluate(basis, X), 
                  w -> (NoTangent(), NoTangent(), _rrule_evaluate(basis, X, w))

                  
function _rrule_evaluate_d(basis::Rn1pBasis, X::AbstractState, 
                           w::AbstractVector, 
                           dRn = evaluate_d(basis.R, norm(_rr(X, basis))))
   rr = _rr(X, basis); r = norm(rr); r̂ = rr/r
   w_r̂ = [ dot(_rr(w, basis), r̂)  for w in w ]
   w2 = [ _rr(w[n], basis) - w_r̂[n] * r̂ for n = 1:length(w) ]
   a = _rrule_evaluate_d(basis.R, r, w_r̂)
   b = sum(dRn .* w2) / r 
   TDX = ACE.dstate_type(a, X)
   return TDX( NamedTuple{(_varsym(basis),)}( (a * r̂ + b,) ) )
end

function rrule(::typeof(evaluate_d), basis::Rn1pBasis, X::AbstractState)
   rr = _rr(X, basis); r = norm(rr); r̂ = rr/r
   dRn_ = evaluate_d(basis.R, r);
   TDX = dstate_type(valtype(basis, X), X)
   dRn = [ TDX( NamedTuple{(_varsym(basis),)}( (dr * r̂,) ) ) 
           for dr in dRn_ ]
   return dRn, 
          w -> (NoTangent(), NoTangent(), _rrule_evaluate_d(basis, X, w, dRn_))
end
