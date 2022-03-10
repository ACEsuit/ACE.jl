

# ------------------ Some different ways to produce an argument 

abstract type StaticGet end 

ACE.evaluate(fval::StaticGet, X) = getval(X, fval)
ACE.evaluate_d(fval::StaticGet, X) = getval_d(X, fval)


struct GetVal{VSYM} <: StaticGet end 

getval(X, ::GetVal{VSYM}) where {VSYM} = getproperty(X, VSYM) 

getval_d(X, ::GetVal{VSYM}) where {VSYM} = 
      DState( NamedTuple{(VSYM,)}( (one(getproperty(X, VSYM)),) ) )

get_symbols(::GetVal{VSYM}) where {VSYM} = (VSYM,)


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

get_symbols(::GetNorm{VSYM}) where {VSYM} = (VSYM,)


write_dict(fval::StaticGet) = Dict("__id__" => "ACE_StaticGet", 
                                   "expr" => string(typeof(fval)) )

read_dict(::Val{:ACE_StaticGet}, D::Dict) = eval( Meta.parse(D["expr"]) )()
