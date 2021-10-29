

import ACE: OneParticleBasis, AbstractState
import ACE.OrthPolys: TransformedPolys

import ForwardDiff
import NamedTupleTools
using NamedTupleTools: namedtuple


@doc raw"""
`struct Scal1pBasis <: OneParticleBasis`

One-particle basis of the form $P_n(x_i)$ for a general scalar, invariant 
input `x`. This type basically just translates the `TransformedPolys` into a valid
one-particle basis.
"""
mutable struct Scal1pBasis{VSYM, VIDX, ISYM, T, TT, TJ} <: OneParticleBasis{T}
   P::TransformedPolys{T, TT, TJ}
   B_pool::VectorPool{T}
   dB_pool::VectorPool{T}   
end


scal1pbasis(varsym::Symbol, idxsym::Symbol, args...; varidx = 1, kwargs...) = 
            Scal1pBasis(varsym, varidx, idxsym,  
                  ACE.OrthPolys.transformed_jacobi(args...; kwargs...))

Scal1pBasis(varsym::Symbol, idxsym::Symbol, P::TransformedPolys{T, TT, TJ}
            ) where {T, TT, TJ} = Scal1pBasis(varsym, 1, idxsym, P)

Scal1pBasis(varsym::Symbol, varidx::Integer, idxsym::Symbol, P::TransformedPolys{T, TT, TJ}
            ) where {T, TT, TJ} = 
      Scal1pBasis{varsym, Int(varidx), idxsym, T, TT, TJ}(P)

Scal1pBasis{VSYM, VIDX, ISYM, T, TT, TJ}(P::TransformedPolys{T, TT, TJ}
            ) where {VSYM, VIDX, ISYM, T, TT, TJ} = 
      Scal1pBasis{VSYM, VIDX, ISYM, T, TT, TJ}(P, VectorPool{T}(), VectorPool{T}())

_varsym(basis::Scal1pBasis{VSYM}) where {VSYM} = VSYM
_varidx(basis::Scal1pBasis{VSYM, VIDX}) where {VSYM, VIDX} = VIDX
_idxsym(basis::Scal1pBasis{VSYM, VIDX, ISYM}) where {VSYM, VIDX, ISYM} = ISYM

_val(X::AbstractState, basis::Scal1pBasis) = 
      getproperty(X, _varsym(basis))[_varidx(basis)]

_val(x::Number, basis::Scal1pBasis) = x

# ---------------------- Implementation of Scal1pBasis


Base.length(basis::Scal1pBasis) = length(basis.P)

get_spec(basis::Scal1pBasis, n::Integer) = NamedTuple{(_idxsym(basis),)}(n)

get_spec(basis::Scal1pBasis) = [ get_spec(basis, i) for i = 1:length(basis) ]

function set_spec!(basis::Scal1pBasis, spec::Vector)      
   # we don't want to set anything, just check that its compatible with the spec 
   # we produce on the fly 
   old_spec = get_spec(basis)
   @assert old_spec == spec 
   return basis 
end

==(P1::Scal1pBasis, P2::Scal1pBasis) = (P1.P == P2.P)

write_dict(basis::Scal1pBasis{T}) where {T} = Dict(
      "__id__" => "ACE_Scal1pBasis",
          "P" => write_dict(basis.P) , 
          "varsym" => string(_varsym(basis)), 
          "varidx" => _varidx(basis), 
          "idxsym" => string(_idxsym(basis)) )

read_dict(::Val{:ACE_Scal1pBasis}, D::Dict) =   
      Scal1pBasis(Symbol(D["varsym"]), Int(D["varidx"]), Symbol(D["idxsym"]), 
                  read_dict(D["P"]))

valtype(basis::Scal1pBasis) = valtype(basis.P)

valtype(basis::Scal1pBasis, cfg::AbstractConfiguration) = 
      valtype(basis, zero(eltype(cfg)))

valtype(basis::Scal1pBasis, X::AbstractState) = 
      valtype(basis.P, _val(X, basis))

gradtype(basis::Scal1pBasis, X::AbstractState) = 
      dstate_type(valtype(basis, X), X)

gradtype(basis::Scal1pBasis, cfg::AbstractConfiguration) = 
      gradtype(basis, zero(eltype(cfg)))

argsyms(basis::Scal1pBasis) = ( _varsym(basis), )

symbols(basis::Scal1pBasis) = [ _idxsym(basis) ]

indexrange(basis::Scal1pBasis) = NamedTuple{(_idxsym(basis), )}((1:length(basis),))

_getidx(b, basis::Scal1pBasis) = b[_idxsym(basis) ]

isadmissible(b, basis::Scal1pBasis) = (1 <= _getidx(b, basis) <= length(basis))

degree(b, basis::Scal1pBasis) = _getidx(b, basis) - 1

get_index(basis::Scal1pBasis, b) = _getidx(b, basis)

# TODO: need better structure to support this ... 
rand_radial(basis::Scal1pBasis) = rand_radial(basis.P)

# ---------------------------  Evaluation code
#

evaluate!(B, basis::Scal1pBasis, x::Number) =
      evaluate!(B, basis.P, x)

evaluate!(B, basis::Scal1pBasis, X::AbstractState) =
      evaluate!(B, basis.P, _val(X, basis))

"""
returns an `SVector{N}` of the form `x * e_I` where `e_I` is the Ith canonical basis vector.
"""
@generated function __e(::SVector{N}, ::Val{I}, x::T) where {N, I, T}
   code = "SA["
   for i = 1:N 
      if i == I
         code *= "x,"
      else 
         code *= "0,"
      end
   end
   code *= "]"
   quote 
      $( Meta.parse(code) )
   end
end

__e(::Number, ::Any, x) = x


function _scal1pbasis_grad(TDX::Type, basis::Scal1pBasis, gval)
   gval_tdx = __e( getproperty(zero(TDX), _varsym(basis)), 
                   Val(_varidx(basis)), 
                   gval )
   return TDX( NamedTuple{(_varsym(basis),)}((gval_tdx,)) )
end


function evaluate_d!(dB, basis::Scal1pBasis, X::AbstractState)
   TDX = eltype(dB)
   x = _val(X, basis)
   dP = acquire_dB!(basis.P, x)
   evaluate_d!(dP, basis.P, x)
   for n = 1:length(basis)
      dB[n] = _scal1pbasis_grad(TDX, basis, dP[n])
   end
   release_dB!(basis.P, dP)
   return dB
end

function evaluate_ed!(B, dB, basis::Scal1pBasis, X::AbstractState)
   TDX = eltype(dB)
   x = _val(X, basis)
   dP = acquire_dB!(basis.P, x)
   evaluate!(B, basis.P, x)
   evaluate_d!(dP, basis.P, x)
   for n = 1:length(basis)
      dB[n] = _scal1pbasis_grad(TDX, basis, dP[n])
   end
   release_dB!(basis.P, dP)
   return B, dB
end


# this one we probably only need for training so can relax the efficiency a bit 
function evaluate_dd(basis::Scal1pBasis, X::AbstractState) 
   ddP = ForwardDiff.derivative(x -> evaluate_d(basis, _val(X, basis)))
   TDX = gradtype(basis, X)
   return _scal1pbasis_grad.(Ref(TDX), Ref(basis), ddP_n)
end



# -------------- AD codes 

import ChainRules: rrule, ZeroTangent, NoTangent

function _rrule_evaluate(basis::Scal1pBasis, X::AbstractState, 
                         w::AbstractVector{<: Number})
   @assert _varidx(basis) == 1
   x = _val(X, basis)
   a = _rrule_evaluate(basis.P, x, real.(w))
   TDX = ACE.dstate_type(a, X)
   return TDX( NamedTuple{(_varsym(basis),)}( (a,) ) )
end

rrule(::typeof(evaluate), basis::Scal1pBasis, X::AbstractState) = 
                  evaluate(basis, X), 
                  w -> (NoTangent(), NoTangent(), _rrule_evaluate(basis, X, w))

             
                  
function _rrule_evaluate_d(basis::Scal1pBasis, X::AbstractState, 
                           w::AbstractVector)
   @assert _varidx(basis) == 1
   x = _val(X, basis)
   w1 = [ _val(w, basis) for w in w ]
   a = _rrule_evaluate_d(basis.P, x, w1)
   TDX = ACE.dstate_type(a, X)
   return TDX( NamedTuple{(_varsym(basis),)}( (a,) ) )
end

function rrule(::typeof(evaluate_d), basis::Scal1pBasis, X::AbstractState)
   @assert _varidx(basis) == 1
   x = _val(X, basis)
   dB_ = evaluate_d(basis.P, x)
   TDX = dstate_type(valtype(basis, X), X)
   dB = [ TDX( NamedTuple{(_varsym(basis),)}( (dx,) ) )  for dx in dB_ ]
   return dB, 
          w -> (NoTangent(), NoTangent(), _rrule_evaluate_d(basis, X, w))
end
