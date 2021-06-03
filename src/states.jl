

export ACEConfig, PositionState

using ACE, StaticArrays, NamedTupleTools

import Base: *, +, -, zero, rand, randn, show, promote_rule, rtoldefault, 
       isapprox, getproperty, real 
import LinearAlgebra: norm, promote_leaf_eltypes

abstract type XState{SYMS, TT} <: ACE.AbstractState end 

struct State{SYMS, TT} <: XState{SYMS, TT}
   x::NamedTuple{SYMS, TT}

   State{SYMS, TT}(t::NamedTuple{SYMS1, TT1}) where {SYMS, SYMS1, TT, TT1} = 
      ( SYMS == SYMS1 ? new{SYMS, TT1}(t) 
                      : State{SYMS, TT}( merge( _x(zero(State{SYMS, TT})), t ) ) )

end

State(t::NamedTuple{SYMS, TT}) where {SYMS, TT} = State{SYMS, TT}(t)
State{SYMS}(t::NamedTuple{SYMS1, TT}) where {SYMS, SYMS1, TT} = State{SYMS, TT}(t)

State(; kwargs...) = State(NamedTuple(kwargs))
State{SYMS}(; kwargs...) where {SYMS} = State{SYMS}(NamedTuple(kwargs))
State{SYMS, TT}(; kwargs...) where {SYMS, TT} = State{SYMS, TT}(NamedTuple(kwargs))

struct DState{SYMS, TT} <: XState{SYMS, TT}
   x::NamedTuple{SYMS, TT}

   DState{SYMS, TT}(t::NamedTuple{SYMS1, TT1}) where {SYMS, SYMS1, TT, TT1} = 
      ( SYMS == SYMS1 ? new{SYMS, TT1}(t) 
                      : DState{SYMS, TT}( merge( _x(zero(State{SYMS, TT})), t ) ) )

end

DState(t::NamedTuple{SYMS, TT}) where {SYMS, TT} = DState{SYMS, TT}(t)

DState(; kwargs...) = DState(kwargs)

_x(X::XState) = getfield(X, :x)
getproperty(X::XState, sym::Symbol) = getproperty(_x(X), sym)

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
_showsym(X::DState) = "'"

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

isapprox(X1::TX, X2::TX, args...; kwargs...
         ) where {TX <: XState{SYMS}} where {SYMS} = 
   all( isapprox( getproperty(_x(X1), sym), getproperty(_x(X2), sym), 
                  args...; kwargs...) for sym in SYMS )


# ----- Implementation of a Position State, as a basic example 
PositionState{T} = State{(:rr,), Tuple{SVector{3, T}}}

promote_rule(::Union{Type{S}, Type{PositionState{S}}}, 
             ::Type{PositionState{T}}) where {S, T} = 
      PositionState{promote_type(S, T)}

# some special functionality for PositionState 
*(A::AbstractMatrix, X::TX) where {TX <: PositionState} = TX( (rr = A * X.rr,) )
+(X::TX, u::SVector{3}) where {TX <: PositionState} = TX( (rr = X.rr + u,) )

real(X::PositionState{T}) where {T} = 
            PositionState{real(T)}( (rr = real.(X.rr), ) )

# ------------------ Basic Configurations Code 

struct ACEConfig{STT} <: AbstractConfiguration
   Xs::Vector{STT}   # list of states
end

# --- iterator to go through all states in an abstract configuration assuming
#     that the states are stored in cfg.Xs

Base.iterate(cfg::AbstractConfiguration) =
   length(cfg.Xs) == 0 ? nothing : (cfg.Xs[1], 1)

Base.iterate(cfg::AbstractConfiguration, i::Integer) =
   length(cfg.Xs) == i ? nothing : (cfg.Xs[i+1], i+1)

Base.length(cfg::AbstractConfiguration) = length(cfg.Xs)

Base.eltype(cfg::AbstractConfiguration) = eltype(cfg.Xs)