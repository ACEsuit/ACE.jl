

export State, PositionState

using StaticArrays, NamedTupleTools

import Base: *, +, -, zero, rand, randn, show, promote_rule, rtoldefault, 
       isapprox, getproperty, real 

import LinearAlgebra: norm, promote_leaf_eltypes


abstract type XState{NT <: NamedTuple} <: AbstractState end 

"""
`struct State` the main type for states of input variables (particles). 
This type is intended only for storing of information but no arithmetic 
should be performed on it. For the latter, we have the DState.
"""
struct State{NT <: NamedTuple} <: XState{NT}
   x::NT

   # standard constructor - SYMS are the same 
   State{NT}(t::NT1) where {NT <: NamedTuple{SYMS}, NT1 <: NamedTuple{SYMS}} where {SYMS} = 
         new{NT1}(t)

   # if SYMS are not the same we automatically merge them
   State{NT}(t::NT1) where {NT <: NamedTuple, NT1 <: NamedTuple} = 
         State( merge( _x(zero(State{NT})), t ) )
end

"""
`struct DState`: A `State`-like variable but acting like vector with arithmetic 
operations defined on it, while `State` acts more like a fixed object that cannot 
be manipulated. The main application of `DState` is as a derivative of a 
`State`; see also `dstate_type`, ``
"""
struct DState{NT <: NamedTuple} <: XState{NT}
   x::NT

   # standard constructor - SYMS are the same 
   DState{NT}(t::NT1) where {NT <: NamedTuple{SYMS}, NT1 <: NamedTuple{SYMS}} where {SYMS} = 
         new{NT1}(t)

   # if SYMS are not the same we automatically merge them
   DState{NT}(t::NT1) where {NT <: NamedTuple, NT1 <: NamedTuple} = 
         DState( merge( _x(zero(DState{NT})), t ) )
end

# the two standard outward facing constructors
State(nt::NT) where {NT <: NamedTuple} = State{NT}(nt)
State(; kwargs...) = State(NamedTuple(kwargs))

# some variations on the above... which might not be needed anymore 
State{NT}(; kwargs...) where {NT <: NamedTuple} = State{NT}(NamedTuple(kwargs))


# accessing X.nt and the fields of X.nt 
# this relies heavily on constant propagation 
_x(X::XState) = getfield(X, :x)
getproperty(X::XState, sym::Symbol) = getproperty(_x(X), sym)

# ----------- printing / output 

const _showdigits = Ref{Int64}(2)

"""
change how many digits are printed in the ACE States
"""
function showdigits!(n::Integer)
   _showdigits[] = n 
end 

_2str(x) = string(x)
_2str(x::AbstractFloat) = "[$(round(x, digits=_showdigits[]))]"
_2str(x::Complex) = "[$(round(x, digits=_showdigits[]))]"
_2str(x::SVector{N, <: AbstractFloat}) where {N} = string(round.(x, digits=_showdigits[]))
_2str(x::SVector{N, <: Complex}) where {N} = string(round.(x, digits=_showdigits[]))[11:end]

_showsym(X::State) = ""
_showsym(X::DState) = "′"

function show(io::IO, X::XState) 
   str = prod( "$(sym):$(_2str(getproperty(_x(X), sym))), " 
               for sym in keys(_x(X)) )
   print(io,  "⟨" * str[1:end-2] * "⟩" * _showsym(X))
end

# ----------- some basic manipulations 

# extract the symbols and the types 
_syms(X::XState) = _syms(typeof(X))
_syms(::Type{<: XState{NamedTuple{SYMS, TT}}}) where {SYMS, TT} = SYMS

_tt(X::XState) = _tt(typeof(X))
_tt(::Type{<: XState{NamedTuple{SYMS, TT}}}) where {SYMS, TT} = TT

_symstt(X::XState) = _symstt(typeof(X))
_symstt(::Type{<: XState{NamedTuple{SYMS, TT}}}) where {SYMS, TT} = SYMS, TT 

# which properties are continuous 

const CTSTT = Union{AbstractFloat, Complex{<: AbstractFloat},
                    SVector{N, <: AbstractFloat}, 
                    SVector{N, <: Complex}} where {N}

"""
Find the indices of continuous properties, and return the 
indices as well as the symbols and types
"""
_findcts(X::TX) where {TX <: XState} = _findcts(TX)

function _findcts(TX::Type{<: XState})
   SYMS, TT = _symstt(TX)
   icts = findall(T -> T <: CTSTT, TT.types)
end

_ctssyms(X::TX) where {TX <: XState} = _ctssyms(TX)

_ctssyms(TX::Type{<: XState}) = _syms(TX)[_findcts(TX)]


## ----- DState constructors

DState(t::NT) where {NT <: NamedTuple} = DState{NT}(t)

DState(; kwargs...) = DState(NamedTuple(kwargs))

DState(X::TX) where {TX <: State} = 
      (dstate_type(X))( select(_x(X), _ctssyms(X)) )


      
"""
convert a State to a corresponding DState 
(basically just remove the discrete variables)
"""
dstate_type(X::DState) = typeof(X)

@generated function dstate_type(X::TX)  where {TX <: State}
   CSYMS = _ctssyms(TX) 
   quote
      typeof( DState( select(_x(X), $CSYMS) ) )
   end
end
      
# the next variant of dstate_type is used to potentially extend 
# from real states to complex dstates. 

_mypromrl(T::Type{<: Number}, S::Type{<: Number}) = 
      promote_type(T, S)
_mypromrl(T::Type{<: Number}, ::Type{<: SVector{N, P}}) where {N, P} = 
      SVector{N, promote_type(T, P)}
_mypromrl(::Type{<: SVector{N, P}}, T::Type{<: Number}) where {N, P} = 
      promote_type(T, P)
_mypromrl(T::Type{<: SVector{N, P1}}, ::Type{<: SVector{N, P2}}) where {N, P1, P2} = 
      SVector{N, promote_type(P1, P2)}

@generated function dstate_type(x::S, X::TX) where {S, TX <: XState}
   SYMS, TT = _symstt(TX)
   icts = _findcts(TX)
   CSYMS = SYMS[icts]
   CTT = [ _mypromrl(S, TT.types[i]) for i in icts ]
   CTTstr = "Tuple{" * "$(tuple(CTT...))"[2:end-1] * "}"
   quote
      $(Meta.parse( "DState{NamedTuple{$(CSYMS), $CTTstr}}" ))
   end
end

dstate_type(S::Type, X::XState) = dstate_type(zero(S), X)

## ---------- explicit real/complex conversion 
# this feels a bit like a hack but might be unavoidable; 
# real, complex goes to _ace_real, _ace_complex, which is then applied 
# in only slightly non-standard fashion recursively to the states


for f in (:real, :imag, :complex, )
   face = Symbol("_ace_$f")
   eval(quote
      import Base: $f
      $face(x::Number) = $f(x)
      $face(x::StaticArrays.StaticArray) = $f.(x)
      function $f(X::TDX) where {TDX <: DState}
         SYMS = _syms(TDX)
         vals = ntuple(i -> $face(getproperty(X, SYMS[i])), length(SYMS))
         return TDX( NamedTuple{SYMS}(vals) )
      end
   end)
end

for f in (:real, :complex, )
   face = Symbol("_ace_$f")
   eval(quote
      $f(TDX::Type{<: DState}) = typeof( $f(zero(TDX)) )
      # import Base: $f
      # $face(x::Type{<: Number}) = $f(x)
      # $face(x::SVector{N, T}) where {N, T} = SVector{N, $f(T)}
      # function $f(TDX::Type{<: DState})
      #    SYMS, TT = _symstt(TDX)
      #    TT1 = ntuple(i -> $face(TT[i]), length(SYMS))
      #    vals = zero.(TT1)
      #    return typeof(TDX( NamedTuple{SYMS}(vals) )
      # end
   end)
end



for f in (:rand, :randn, :zero)
   face = Symbol("_ace_$f")
   eval( quote 
      import Base: $f 
      $face(T::Type) = $f(T) 
      $face(x::Union{Number, AbstractArray}) = $f(typeof(x))

      function $f(x::Union{TX, Type{TX}}) where {TX <: XState}
         SYMS, TT = _symstt(x)
         vals = ntuple(i -> $face(TT.types[i]), length(SYMS))
         return TX( NamedTuple{SYMS}( vals ) )
      end
   end )
end

# an extra for symbols, this is a bit questionable; why do we even need it?
_ace_zero(::Union{Symbol, Type{Symbol}}) = :O


## ----------- Some arithmetic operations 

# binary operations 

import Base: +, -


for f in (:+, :-, )
   eval( quote 
      function $f(X1::TX1, X2::TX2) where {TX1 <: XState, TX2 <: XState}
         SYMS = _syms(TX1)
         @assert SYMS == _syms(TX2)
         vals = ntuple( i -> $f( getproperty(_x(X1), SYMS[i]), 
                                 getproperty(_x(X2), SYMS[i]) ), length(SYMS) )
         return TX1( NamedTuple{SYMS}(vals) )
      end
   end )
end

# multiplication with a scalar 
function *(X1::TX, a::Number) where {TX <: XState}
   SYMS = _syms(TX)
   vals = ntuple( i -> *( getproperty(_x(X1), SYMS[i]), a ), length(SYMS) )
   return TX( NamedTuple{SYMS}(vals) )
end

*(a::Number, X1::XState) = *(X1, a)

*(aa::SVector{N, <: Number}, X1::XState) where {N} = aa .* Ref(X1)
promote_rule(::Type{SVector{N, T}}, ::Type{TX}) where {N, T <: Number, TX <: XState} = 
      SVector{N, promote_type(T, TX)}

# unary 
import Base: - 

for f in (:-, )
   eval(quote
      function $f(X::TX) where {TX <: XState}
         SYMS = _syms(TX)
         vals = ntuple( i -> $f( getproperty(_x(X), SYMS[i]) ), length(SYMS) )
         return TX( NamedTuple{SYMS}(vals) )
      end
   end)
end


# reduction to scalar 

import LinearAlgebra: dot 
import Base: isapprox



for (f, g) in ( (:dot, :sum), (:isapprox, :all) )  # (:contract, :sum), 
   eval( quote 
      function $f(X1::TX1, X2::TX2) where {TX1 <: XState, TX2 <: XState}
         SYMS = _syms(TX1)
         @assert SYMS == _syms(TX2)
         return $g( $f( getproperty(_x(X1), sym), 
                         getproperty(_x(X2), sym) )   for sym in SYMS)
      end
   end )
end

@generated function contract(X1::TX1, X2::TX2) where {TX1 <: XState, TX2 <: XState}
   SYMS = _syms(TX1)
   @assert SYMS == _syms(TX2)
   code = "contract(X1.$(SYMS[1]), X2.$(SYMS[1]))"
   for sym in SYMS[2:end]
      code *= " + contract(X1.$sym, X2.$sym)"
   end
   return quote 
      $(Meta.parse(code))
   end
end

contract(X1::Number, X2::XState) = X1 * X2 
contract(X2::XState, X1::Number) = X1 * X2 


import LinearAlgebra: norm 

for (f, g) in ((:norm, :norm), (:sumsq, :sum), (:normsq, :sum) )
   eval( quote 
      function $f(X::TX) where {TX <: XState}
         SYMS = _syms(TX)
         vals = ntuple( i -> $f( getproperty(_x(X), SYMS[i]) ),  length(SYMS))
         return $g(vals)
      end
   end )
end



## --------  not clear where needed; or deleted functionality 

function promote_leaf_eltypes(X::TX) where {TX <: XState} 
   SYMS = _syms(TX)
   promote_type( ntuple(i -> promote_leaf_eltypes(getproperty(_x(X), SYMS[i])), length(SYMS))... )
end


## ----- Implementation of a Position State, as a basic example 

PositionState{T} = State{NamedTuple{(:rr,), Tuple{SVector{3, T}}}}

PositionState(r::AbstractVector{T}) where {T <: AbstractFloat} = 
      (@assert length(r) == 3; PositionState{T}(; rr = SVector{3, T}(r)))

promote_rule(::Union{Type{S}, Type{PositionState{S}}}, 
             ::Type{PositionState{T}}) where {S, T} = 
      PositionState{promote_type(S, T)}

# some special functionality for PositionState, mostly needed for testing  
*(A::AbstractMatrix, X::TX) where {TX <: PositionState} = TX( (rr = A * X.rr,) )
+(X::TX, u::StaticVector{3}) where {TX <: PositionState} = TX( (rr = X.rr + u,) )




# ------------------ Basic Configurations Code 
# TODO: get rid of this?  

import ACEbase: AbstractConfiguration
export ACEConfig 

"""
`struct ACEConfig`: The canonical implementation of an `AbstractConfiguration`. 
Just wraps a `Vector{<: AbstractState}`
"""
struct ACEConfig{STT} <: AbstractConfiguration
   Xs::Vector{STT}   # list of states
end

Base.eltype(cfg::ACEConfig) = eltype(cfg.Xs)

Base.iterate(cfg::ACEConfig, args...) = iterate(cfg.Xs, args...)

Base.length(cfg::ACEConfig) = length(cfg.Xs)

# ---------------- AD code 

# TODO: check whether this is still needed 
# this function makes sure that gradients w.r.t. a State become a DState 
function rrule(::typeof(getproperty), X::XState, sym::Symbol) 
   val = getproperty(X, sym)
   return val, w -> ( NoTangent(), 
                      dstate_type(w[1], X)( NamedTuple{(sym,)}((w,)) ), 
                      NoTangent() )
end




