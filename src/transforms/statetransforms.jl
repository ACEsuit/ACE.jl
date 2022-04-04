using StaticArrays
import ACE 
import ACE: evaluate, evaluate_d, evaluate_dd, valtype, gradtype, 
            write_dict, read_dict, 
            DState, 
            rrule_evaluate!, frule_evaluate! 

using LinearAlgebra: I, norm  

# TODO: 
#   - retire GetNorm
#   - polish GetVal, and expand to GetVali 
#   - polish the frule and rrule implementations 

# ------------------ Some different ways to produce an argument 

abstract type StateTransform end 
abstract type StaticGet <: StateTransform end 

ACE.evaluate(fval::StaticGet, X) = getval(X, fval)
ACE.evaluate_d(fval::StaticGet, X) = getval_d(X, fval)

valtype(fval::StaticGet, X) = typeof(evaluate(fval, X))
gradtype(fval::StaticGet, X) = typeof(evaluate_d(fval, X))


struct GetVal{VSYM} <: StaticGet end 

getval(X, ::GetVal{VSYM}) where {VSYM} = getproperty(X, VSYM) 

_one(x::Number) = one(x)
_one(x::SVector{3, T}) where {T}  = SMatrix{3, 3, T}(I)

getval_d(X, ::GetVal{VSYM}) where {VSYM} = 
      DState( NamedTuple{(VSYM,)}( (_one(getproperty(X, VSYM)),) ) )

get_symbols(::GetVal{VSYM}) where {VSYM} = (VSYM,)

function rrule_evaluate!(dB, dP, ::GetVal{VSYM}, X) where {VSYM}
   x = getproperty(X, VSYM)
   TDX = eltype(dB)
   for n = 1:length(dB)
      dB[n] = TDX(  DState( NamedTuple{(VSYM,)}( ( dP[n], ) ) ) )
   end
   return dB 
end

grad_type_dP(TDP, ::GetVal{VSYM}, X) where {VSYM} = 
      typeof(DState( NamedTuple{(VSYM,)}( (zero(TDP),) ) ))


function rrule_evaluate(dP, ::GetVal{VSYM}, X) where {VSYM}
   x = getproperty(X, VSYM)
   return [ DState( NamedTuple{(VSYM,)}( ( dP[n], ) ) )
            for n = 1:length(dP) ]
end
      
# TODO - this is incomplete for now 
# struct GetVali{VSYM, IND} <: StaticGet end 
# getval(X, ::GetVali{VSYM, IND}) where {VSYM, IND} = getproperty(X, VSYM)[IND]
# getval_d(X, ::GetVali{VSYM, IND}) where {VSYM, IND} = __e(getproperty(X, VSYM), Val{IND}())


struct GetNorm{VSYM} <: StaticGet end 

getval(X, ::GetNorm{VSYM}) where {VSYM} = norm(getproperty(X, VSYM))

function getval_d(X, ::GetNorm{VSYM}) where {VSYM}
   x = getproperty(X, VSYM)
   return DState( NamedTuple{(VSYM,)}( (x/norm(x),) ) )
end 

function evaluate_dd(::GetNorm{VSYM}, X) where {VSYM}
   𝐫 = getproperty(X,VSYM)
   r = norm(𝐫)
   𝐫̂ = 𝐫 / r
   ddx = (I - 𝐫̂ * 𝐫̂') / r
   return DState( NamedTuple{(VSYM,)}( (ddx,) ) )
end 


get_symbols(::GetNorm{VSYM}) where {VSYM} = (VSYM,)


write_dict(fval::StaticGet) = Dict("__id__" => "ACE_StaticGet", 
                                   "expr" => string(typeof(fval)) )

read_dict(::Val{:ACE_StaticGet}, D::Dict) = eval( Meta.parse(D["expr"]) )()


function rrule_evaluate!(dB, dP, ::GetNorm{VSYM}, X) where {VSYM}
   x = getproperty(X, VSYM)
   dx = x/norm(x)
   TDX = eltype(dB)
   for n = 1:length(dB)
      dB[n] = TDX(  DState( NamedTuple{(VSYM,)}( ( dx * dP[n], ) ) ) )
   end
   return dB 
end

grad_type_dP(TDP, ::GetNorm{VSYM}, X) where {VSYM} = 
      typeof(DState( NamedTuple{(VSYM,)}( (zero(SVector{3,TDP}),) ) ))

