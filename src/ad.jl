
function rrule_evaluate end 

function frule_evaluate end 


struct SChain{TT}
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

@generated function evaluate(chain::SChain{TT}, X) where {TT}
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


# function evaluate_ed(chain::SChain{TT}, X) where {TT} 
#    LEN = length(chain.F)
#    Xi = evaluate(chain.F[1], X)
#    dFi = evaluate_d(chain.F[1], X)
#    for i = 2:LEN
#       Xi, dFi = frule_evaluate(chain.F[i], Xi, dFi)
#    end
#    return Xi, dFi 
# end

function evaluate_ed(chain::SChain{TT}, X) where {TT} 
   LEN = length(chain.F)
   Xi = evaluate(chain.F[1], X)
   dFi = evaluate_d(chain.F[1], X)
   for i = 2:LEN
      dFi = evaluate_d(chain.F[i], Xi)
      Xi = evaluate(chain.F[i], Xi)
   end
   return Xi, dFi 
end


##
nothing 

##

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