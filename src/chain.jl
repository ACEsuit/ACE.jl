abstract type AbstractSChain{TT} end 
struct SChain{TT} <: AbstractSChain{TT}
   F::TT
end

struct TypedChain{TT, IN, OUT} <: AbstractSChain{TT}
   F::TT 
end 


# construct a chain recursively 
chain(F1, F2, args...) = chain( chain(F1, F2), args... )
# for most arguments, just form a tuple 
chain(F1, F2) = SChain( (F1, F2) )
# if one of them is a chain already, then combine into a single long chain 
chain(F1::SChain, F2) = SChain( tuple(F1.F..., F2) )
chain(F1, F2::SChain) = SChain( tuple(F1, F2.F...) )
chain(F1::SChain, F2::SChain) = chain( tuple(F1.F..., F2.F...) )

Base.length(c::SChain) = length(c.F)

@generated function evaluate(chain::AbstractSChain{TT}, X) where {TT}
   LEN = length(TT.types)
   code = Expr[]  
   push!(code, :(X_0 = X))
   for l = 1:LEN 
      push!(code, Meta.parse("F_$l = chain.F[$l]"))
      push!(code, Meta.parse("X_$l = evaluate(F_$l, X_$(l-1))"))
      push!(code, Meta.parse("release!(X_$(l-1))"))
   end
   push!(code, Meta.parse("return X_$LEN"))
   return Expr(:block, code...)
end

# TODO: 
# - replace with an frule and then wrap that into an evaluate_ed 
# - generated function to make this fast 
# - implement the rrule 

@generated function evaluate_ed(chain::AbstractSChain{TT}, X) where {TT}
   LEN = length(TT.types)
   code = Expr[]  
   push!(code, :(X_0 = X))
   push!(code, Meta.parse("X_1, dF_1 = evaluate_ed(chain.F[1], X_0)"))
   for l = 2:LEN 
      push!(code, Meta.parse("F_$l = chain.F[$l]"))
      push!(code, Meta.parse("X_$l, dF_$l = frule_evaluate(F_$l, X_$(l-1), dF_$(l-1))"))
      push!(code, Meta.parse("release!(X_$(l-1))"))
   end
   push!(code, Meta.parse("return X_$LEN, dF_$LEN"))
   return Expr(:block, code...)
end

# function evaluate_ed(chain::AbstractSChain{TT}, X) where {TT} 
#    LEN = length(chain.F)
#    Xi = evaluate(chain.F[1], X)
#    dFi = evaluate_d(chain.F[1], X)
#    for i = 2:LEN
#       Xi, dFi = frule_evaluate(chain.F[i], Xi, dFi)
#    end
#    return Xi, dFi 
# end


evaluate_d(chain::AbstractSChain, X) = evaluate_ed(chain, X)[2]


# TODO: This still needs sorting out ... 
#       maybe we can no kill this? 

valtype(chain::SChain) = valtype(chain.F[end])

valtype(chain::SChain, x) = valtype(chain)

gradtype(chain::SChain) = gradtype(chain.F[end])

# function gradtype(chain::SChain, x) 
#    LEN = length(chain.F)
#    Ti = gradtype(chain.F[1], x)

# end


valtype(chain::TypedChain{TT, IN, OUT}, args...) where {TT, IN, OUT} = OUT

gradtype(chain::TypedChain{TT, IN, OUT}, args...) where {TT, IN, OUT} = _gradtype(OUT, IN)

_gradtype(::Type{Vector{T1}}, ::Type{T2}) where {T1 <: Number, T2 <: Number} = 
      Vector{promote_type(T1, T2)}

_gradtype(::Type{Vector{T1}}, ::Type{SVector{N, T2}}) where {N, T1 <: Number, T2 <: Number} = 
      Vector{SVector{N, promote_type(T1, T2)}}


## 

import Base: == 

==(ch1::SChain, ch2::SChain) = 
      all( F1==F2 for (F1, F2) in zip(ch1.F, ch2.F) )

write_dict(chain::SChain) = Dict(
            "__id__" => "ACE_SChain", 
            "F" => write_dict.(chain.F)
         )

read_dict(::Val{:ACE_SChain}, D::Dict) = 
         SChain(tuple( read_dict.(D["F"])... ))

##

#=

# Experimental AD codes

import ChainRulesCore, ChainRules
import ChainRulesCore: rrule, NoTangent 

struct EVAL{TM, TX}
   m::TM 
   X::TX
end

struct EVAL_D{TM, TX}
   m::TM 
   X::TX
end


function (y::EVAL)(params)
   set_params!(y.m, params)
   return evaluate(y.m, y.X).val
end

function rrule(y::EVAL, params)
   set_params!(y.m, params)
   val = evaluate(y.m, y.X).val
   adj = dp -> ( NoTangent(), dp * getproperty.(grad_params(y.m, y.X), :val)) 
   return val, adj
end


function (y::EVAL_D)(params)
   set_params!(y.m, params)
   return grad_config(y.m, y.X)
end


function rrule(y::EVAL_D, params)
   set_params!(y.m, params)
   val = grad_config(y.m, y.X)
   adj = dp -> ( NoTangent(), adjoint_EVAL_D(y.m, y.X, dp) )
   return val, adj
end

=#