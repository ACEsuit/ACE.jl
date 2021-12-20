

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
         State{SYMS, TT}( merge( _x(zero(State{SYMS, TT})), t ) )
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
         DState{SYMS, TT}( merge( _x(zero(State{SYMS, TT})), t ) )
end

# the two standard outward facing constructors
State(nt::NT) where {NT <: NamedTuple} = State{NT}(nt)
State(; kwargs...) = State(NamedTuple(kwargs))

# # some variations on the above...
# State{SYMS}(t::NamedTuple{SYMS1, TT}) where {SYMS, SYMS1, TT} = State{SYMS, TT}(t)
# State{SYMS}(; kwargs...) where {SYMS} = State{SYMS}(NamedTuple(kwargs))
# State{SYMS, TT}(; kwargs...) where {SYMS, TT} = State{SYMS, TT}(NamedTuple(kwargs))


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

# ----- DState constructors

DState(t::NT) where {NT <: NamedTuple} = DState{NT}(t)

DState(; kwargs...) = DState(NamedTuple(kwargs))

DState(X::TX) where {TX <: State} = 
      (dstate_type(X))( select(_x(X), _ctssyms(X)) )

      

# -------------- Some weird stuff I don't remember 
#   looks like a hack and should probably be looked at 
#   very carefully 

# _mypromrl(T::Type{<: Number}, S::Type{<: Number}) = 
#       promote_type(T, S)
# _mypromrl(T::Type{<: Number}, ::Type{<: SVector{N, P}}) where {N, P} = 
#       SVector{N, promote_type(T, P)}
# _mypromrl(::Type{<: SVector{N, P}}, T::Type{<: Number}) where {N, P} = 
#       promote_type(T, P)
# _mypromrl(T::Type{<: SVector{N, P1}}, ::Type{<: SVector{N, P2}}) where {N, P1, P2} = 
#       SVector{N, promote_type(P1, P2)}

# @generated function dstate_type(x::S, X::ACE.XState{SYMS, TT}
#                                 ) where {S, SYMS, TT}
#    syms2 = Symbol[] 
#    tt2 = DataType[]
#    for i = 1:length(SYMS)
#       if TT.types[i] <: CTSTT
#          push!(syms2, SYMS[i])
#          push!(tt2, _mypromrl(S, TT.types[i]))
#       end
#    end
#    SYMS2 = tuple(syms2...)
#    TT2 = "Tuple{" * 
#             "$(tuple(tt2...))"[2:end-1] * "}"
#    DTX = Meta.parse( "DState{$(SYMS2), $TT2}" )
#    quote
#       $DTX 
#    end
# end

# dstate_type(S::Type, X::ACE.State) = dstate_type(zero(S), X)




# _myrl(x::Number) = real(x)
# _myrl(x::StaticArrays.StaticArray) = real.(x)
# Base.real(X::TDX) where {TDX <: DState{SYMS}} where {SYMS} = 
#       TDX( NamedTuple{SYMS}( ntuple(i -> _myrl(getproperty(X, SYMS[i])), length(SYMS)) ) )

# _myim(x::Number) = imag(x)
# _myim(x::SVector) = imag.(x)
# Base.imag(X::TDX) where {TDX <: DState{SYMS}} where {SYMS} = 
#       TDX( NamedTuple{SYMS}( ntuple(i -> _myim(getproperty(X, SYMS[i])), length(SYMS)) ) )
    
# _mycplx(x::Number) = complex(x)
# _mycplx(x::SVector) = complex.(x)
# Base.complex(X::TDX) where {TDX <: DState{SYMS}} where {SYMS} =
#       TDX( NamedTuple{SYMS}( ntuple(i -> _mycplx(getproperty(X, SYMS[i])), length(SYMS)) ) )

# Base.complex(::Type{TDX}) where {TDX <: DState{SYMS}} where {SYMS} =
#       typeof( complex( zero(TDX) ) )
 

# function zero(::Union{TX, Type{TX}}) where {TX <: XState{SYMS, TT}} where {SYMS, TT} 
#    vals = ntuple(i -> _ace_zero(TT.types[i]), length(SYMS))
#    return TX( NamedTuple{SYMS}( vals ) )
# end

# _ace_zero(args...) = zero(args...)
# _ace_zero(::Union{Symbol, Type{Symbol}}) = :O


# for f in (:rand, :randn) 
#    eval( quote 
#       function $f(::Union{TX, Type{TX}}) where {TX <: XState{SYMS, TT}} where {SYMS, TT} 
#          vals = ntuple(i -> $f(TT.types[i]), length(SYMS))
#          return TX( NamedTuple{SYMS}( vals ) )
#       end
#    end )
# end


# for f in (:+, :-)
#    eval( quote 
#       function $f(X1::TX1, X2::TX2) where {TX1 <: XState{SYMS}, TX2 <: XState{SYMS}} where {SYMS}
#          vals = ntuple( i -> $f( getproperty(_x(X1), SYMS[i]), 
#                                  getproperty(_x(X2), SYMS[i]) ), length(SYMS) )
#          return TX1( NamedTuple{SYMS}(vals) )
#       end
#    end )
# end

# function -(X::TX) where {TX <: XState{SYMS}} where {SYMS}
#       vals = ntuple( i -> -getproperty(_x(X), SYMS[i]) )
#       return TX( NamedTuple{SYMS}(vals) )
# end

# function *(X1::TX, a::Number) where {TX <: XState{SYMS}} where {SYMS}
#    vals = ntuple( i -> *( getproperty(_x(X1), SYMS[i]), a ), length(SYMS) )
#    return TX( NamedTuple{SYMS}(vals) )
# end

# function *(a::Number, X1::TX) where {TX <: XState{SYMS}} where {SYMS}
#    vals = ntuple( i -> *( getproperty(_x(X1), SYMS[i]), a ), length(SYMS) )
#    return TX( NamedTuple{SYMS}(vals) )
# end


# promote_leaf_eltypes(X::XState{SYMS}) where {SYMS} = 
#    promote_type( ntuple(i -> promote_leaf_eltypes(getproperty(_x(X), SYMS[i])), length(SYMS))... )

# norm(X::XState{SYMS}) where {SYMS} = 
#       sum( norm( getproperty(_x(X), sym) for sym in SYMS )^2 )

# import LinearAlgebra: dot 
# dot(X1::DState{SYMS}, X2::DState{SYMS}) where {SYMS} = 
#    sum( dot( getproperty(_x(X1), sym), getproperty(_x(X2), sym) )
#         for sym in SYMS )

# _contract(X1::DState{SYMS1}, X2::DState{SYMS2}) where {SYMS1, SYMS2} = 
#    sum( sum( getproperty(_x(X1), sym) .* getproperty(_x(X2), sym) )
#               for sym in SYMS1 )

# isapprox(X1::TX, X2::TX, args...; kwargs...
#          ) where {TX <: XState{SYMS}} where {SYMS} = 
#    all( isapprox( getproperty(_x(X1), sym), getproperty(_x(X2), sym), 
#                   args...; kwargs...) for sym in SYMS )


# ----- Implementation of a Position State, as a basic example 

PositionState{T} = typeof( State(rr = zeros(SVector{3, Float64})) )

PositionState(r::AbstractVector{T}) where {T <: AbstractFloat} = 
      (@assert length(r) == 3; PositionState{T}(; rr = SVector{3, T}(r)))

# promote_rule(::Union{Type{S}, Type{PositionState{S}}}, 
#              ::Type{PositionState{T}}) where {S, T} = 
#       PositionState{promote_type(S, T)}

# # some special functionality for PositionState 
# *(A::AbstractMatrix, X::TX) where {TX <: PositionState} = TX( (rr = A * X.rr,) )
# +(X::TX, u::SVector{3}) where {TX <: PositionState} = TX( (rr = X.rr + u,) )

# real(X::PositionState{T}) where {T} = 
#             PositionState{real(T)}( (rr = real.(X.rr), ) )




# ------------------ Basic Configurations Code 
# TODO: get rid of this?  

"""
`struct ACEConfig`: The canonical implementation of an `AbstractConfiguration`. 
Just wraps a `Vector{<: AbstractState}`
"""
struct ACEConfig{STT}
   Xs::Vector{STT}   # list of states
end


# ---------------- AD code 

# this function makes sure that gradients w.r.t. a State become a DState 
function rrule(::typeof(getproperty), X::XState, sym::Symbol) 
   val = getproperty(X, sym)
   return val, w -> ( NoTangent(), 
                      dstate_type(w[1], X)( NamedTuple{(sym,)}((w,)) ), 
                      NoTangent() )
end


