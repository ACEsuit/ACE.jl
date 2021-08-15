

export ACEConfig, PositionState, DACEConfig

using ACE, StaticArrays, NamedTupleTools

import Base: *, +, -, zero, rand, randn, show, promote_rule, rtoldefault, 
       isapprox, getproperty, real 
import LinearAlgebra: norm, promote_leaf_eltypes


abstract type XState{SYMS, TT} <: ACE.AbstractState end 

"""
`struct State` the main type for states of input variables (particles). 
This type is intended only for storing of information but no arithmetic 
should be performed on it. For the latter, we have the DState.
"""
struct State{SYMS, TT} <: XState{SYMS, TT}
   x::NamedTuple{SYMS, TT}

   State{SYMS, TT}(t::NamedTuple{SYMS1, TT1}) where {SYMS, SYMS1, TT, TT1} = 
      ( SYMS == SYMS1 ? new{SYMS, TT1}(t) 
                      : State{SYMS, TT}( merge( _x(zero(State{SYMS, TT})), t ) ) )

end

"""
`struct DState`: A `State`-like variable but acting like vector with arithmetic 
operations defined on it, while `State` acts more like a fixed object that cannot 
be manipulated. The main application of `DState` is as a derivative of a 
`State`; see also `dstate_type`, ``
"""
struct DState{SYMS, TT} <: XState{SYMS, TT}
   x::NamedTuple{SYMS, TT}

   DState{SYMS, TT}(t::NamedTuple{SYMS1, TT1}) where {SYMS, SYMS1, TT, TT1} = 
      ( SYMS == SYMS1 ? new{SYMS, TT1}(t) 
                      : DState{SYMS, TT}( merge( _x(zero(State{SYMS, TT})), t ) ) )

end



State(t::NamedTuple{SYMS, TT}) where {SYMS, TT} = State{SYMS, TT}(t)
State{SYMS}(t::NamedTuple{SYMS1, TT}) where {SYMS, SYMS1, TT} = State{SYMS, TT}(t)

State(; kwargs...) = State(NamedTuple(kwargs))
State{SYMS}(; kwargs...) where {SYMS} = State{SYMS}(NamedTuple(kwargs))
State{SYMS, TT}(; kwargs...) where {SYMS, TT} = State{SYMS, TT}(NamedTuple(kwargs))


const CTSTT = Union{AbstractFloat, Complex{<: AbstractFloat},
               SVector{N, <: AbstractFloat}, 
               SVector{N, <: Complex}} where {N}

dstate_type(X::DState) = typeof(X)

@generated function dstate_type(X::State{SYMS, TT}) where {SYMS, TT}
   syms2 = Symbol[] 
   tt2 = DataType[]
   for i = 1:length(SYMS)
      if TT.types[i] <: CTSTT
         push!(syms2, SYMS[i])
         push!(tt2, TT.types[i])
      end
   end
   SYMS2 = tuple(syms2...)
   TT2 = "Tuple{" * 
            "$(tuple(tt2...))"[2:end-1] * "}"
   DTX = Meta.parse( "DState{$(SYMS2), $TT2}" )
   quote
      $DTX 
   end
end

_mypromrl(T::Type{<: Number}, S::Type{<: Number}) = 
      promote_type(T, S)
_mypromrl(T::Type{<: Number}, ::Type{<: SVector{N, P}}) where {N, P} = 
      SVector{N, promote_type(T, P)}
_mypromrl(::Type{<: SVector{N, P}}, T::Type{<: Number}) where {N, P} = 
      promote_type(T, P)
_mypromrl(T::Type{<: SVector{N, P1}}, ::Type{<: SVector{N, P2}}) where {N, P1, P2} = 
      SVector{N, promote_type(P1, P2)}

@generated function dstate_type(x::S, X::ACE.XState{SYMS, TT}
                                ) where {S, SYMS, TT}
   syms2 = Symbol[] 
   tt2 = DataType[]
   for i = 1:length(SYMS)
      if TT.types[i] <: CTSTT
         push!(syms2, SYMS[i])
         push!(tt2, _mypromrl(S, TT.types[i]))
      end
   end
   SYMS2 = tuple(syms2...)
   TT2 = "Tuple{" * 
            "$(tuple(tt2...))"[2:end-1] * "}"
   DTX = Meta.parse( "DState{$(SYMS2), $TT2}" )
   quote
      $DTX 
   end
end

dstate_type(S::Type, X::ACE.State) = dstate_type(zero(S), X)

@generated function _ctssyms(::NamedTuple{SYMS, TT}) where {SYMS, TT}
   syms2 = Symbol[] 
   for i = 1:length(SYMS)
      if TT.types[i] <: CTSTT
         push!(syms2, SYMS[i])
      end
   end
   SYMS2 = tuple(syms2...)
   quote
      $SYMS2
   end
end



DState(X::TX) where {TX <: State} = 
      (dstate_type(X))( select(_x(X), _ctssyms(_x(X))) )

DState(t::NamedTuple{SYMS, TT}) where {SYMS, TT} = DState{SYMS, TT}(t)

DState(; kwargs...) = DState(NamedTuple(kwargs))

_x(X::XState) = getfield(X, :x)
getproperty(X::XState, sym::Symbol) = getproperty(_x(X), sym)

_myrl(x::Number) = real(x)
_myrl(x::SVector) = real.(x)
Base.real(X::TDX) where {TDX <: DState{SYMS}} where {SYMS} = 
      TDX( NamedTuple{SYMS}( ntuple(i -> _myrl(getproperty(X, SYMS[i])), length(SYMS)) ) )

_myim(x::Number) = imag(x)
_myim(x::SVector) = imag.(x)
Base.imag(X::TDX) where {TDX <: DState{SYMS}} where {SYMS} = 
      TDX( NamedTuple{SYMS}( ntuple(i -> _myim(getproperty(X, SYMS[i])), length(SYMS)) ) )
      

for f in (:zero, :rand, :randn) 
   eval( quote 
      function $f(::Union{TX, Type{TX}}) where {TX <: XState{SYMS, TT}} where {SYMS, TT} 
         vals = ntuple(i -> $f(TT.types[i]), length(SYMS))
         return TX( NamedTuple{SYMS}( vals ) )
      end
   end )
end

const _showdigits = 4
_2str(x) = string(x)
_2str(x::AbstractFloat) = "[$(round(x, digits=_showdigits))]"
_2str(x::Complex) = "[$(round(x, digits=_showdigits))]"
_2str(x::SVector{N, <: AbstractFloat}) where {N} = string(round.(x, digits=_showdigits))
_2str(x::SVector{N, <: Complex}) where {N} = string(round.(x, digits=_showdigits))[11:end]

_showsym(X::State) = ""
_showsym(X::DState) = "′"

show(io::IO, X::XState{SYMS}) where {SYMS} = 
      print(io, "{" * prod( "$(sym)$(_2str(getproperty(_x(X), sym))), " 
                            for sym in SYMS) * "}" * _showsym(X))

for f in (:+, :-)
   eval( quote 
      function $f(X1::TX1, X2::TX2) where {TX1 <: XState{SYMS}, TX2 <: XState{SYMS}} where {SYMS}
         vals = ntuple( i -> $f( getproperty(_x(X1), SYMS[i]), 
                                 getproperty(_x(X2), SYMS[i]) ), length(SYMS) )
         return TX1( NamedTuple{SYMS}(vals) )
      end
   end )
end

function -(X::TX) where {TX <: XState{SYMS}} where {SYMS}
      vals = ntuple( i -> -getproperty(_x(X), SYMS[i]) )
      return TX( NamedTuple{SYMS}(vals) )
end

function *(X1::TX, a::Number) where {TX <: XState{SYMS}} where {SYMS}
   vals = ntuple( i -> *( getproperty(_x(X1), SYMS[i]), a ), length(SYMS) )
   return TX( NamedTuple{SYMS}(vals) )
end

function *(a::Number, X1::TX) where {TX <: XState{SYMS}} where {SYMS}
   vals = ntuple( i -> *( getproperty(_x(X1), SYMS[i]), a ), length(SYMS) )
   return TX( NamedTuple{SYMS}(vals) )
end


promote_leaf_eltypes(X::XState{SYMS}) where {SYMS} = 
   promote_type( ntuple(i -> promote_leaf_eltypes(getproperty(_x(X), SYMS[i])), length(SYMS))... )

norm(X::XState{SYMS}) where {SYMS} = 
      sum( norm( getproperty(_x(X), sym) for sym in SYMS )^2 )

import LinearAlgebra: dot 
dot(X1::DState{SYMS}, X2::DState{SYMS}) where {SYMS} = 
   sum( dot( getproperty(_x(X1), sym), getproperty(_x(X2), sym) )
        for sym in SYMS )

isapprox(X1::TX, X2::TX, args...; kwargs...
         ) where {TX <: XState{SYMS}} where {SYMS} = 
   all( isapprox( getproperty(_x(X1), sym), getproperty(_x(X2), sym), 
                  args...; kwargs...) for sym in SYMS )


# ----- Implementation of a Position State, as a basic example 
PositionState{T} = State{(:rr,), Tuple{SVector{3, T}}}

PositionState(r::AbstractVector{T}) where {T <: AbstractFloat} = 
      (@assert length(r) == 3; PositionState{T}(; rr = SVector{3, T}(r)))

promote_rule(::Union{Type{S}, Type{PositionState{S}}}, 
             ::Type{PositionState{T}}) where {S, T} = 
      PositionState{promote_type(S, T)}

# some special functionality for PositionState 
*(A::AbstractMatrix, X::TX) where {TX <: PositionState} = TX( (rr = A * X.rr,) )
+(X::TX, u::SVector{3}) where {TX <: PositionState} = TX( (rr = X.rr + u,) )

real(X::PositionState{T}) where {T} = 
            PositionState{real(T)}( (rr = real.(X.rr), ) )

# ------------------ Basic Configurations Code 

abstract type XACEConfig <: AbstractConfiguration  end 

"""
`struct ACEConfig`: The canonical implementation of an `AbstractConfiguration`. 
Just wraps a `Vector{<: AbstractState}`
"""
struct ACEConfig{STT} <: XACEConfig
   Xs::Vector{STT}   # list of states
end


struct DACEConfig{STT} <: XACEConfig
   Xs::Vector{STT}   # list of states
end

import Base: * 
*(t::Number, dcfg::DACEConfig) = DACEConfig( t * dcfg.Xs )
+(cfg::ACEConfig, dcfg::DACEConfig) = ACEConfig( cfg.Xs + dcfg.Xs )


# --- iterator to go through all states in an abstract configuration assuming
#     that the states are stored in cfg.Xs

Base.iterate(cfg::XACEConfig) =
   length(cfg.Xs) == 0 ? nothing : (cfg.Xs[1], 1)

Base.iterate(cfg::XACEConfig, i::Integer) =
   length(cfg.Xs) == i ? nothing : (cfg.Xs[i+1], i+1)

Base.length(cfg::XACEConfig) = length(cfg.Xs)

Base.eltype(cfg::XACEConfig) = eltype(cfg.Xs)



# ---------------- AD code 

# this function makes sure that gradients w.r.t. a State become a DState 
function rrule(::typeof(getproperty), X::ACE.XState, sym::Symbol) 
   val = getproperty(X, sym)
   return val, w -> ( NoTangent(), 
                      dstate_type(w[1], X)( NamedTuple{(sym,)}((w,)) ), 
                      NoTangent() )
end


