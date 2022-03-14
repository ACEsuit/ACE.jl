
struct SChain{TC}
   F:Tuple{TC}
end


# construct a chain recursively 
chain(F1, F2, args...) = chain( chain(F1, F2), chain(args...) )
# for most arguments, just form a tuple 
chain(F1, F2) = SChain( (F1, F2) )
# if one of them is a chain already, then combine into a single long chain 
chain(F1::SChain, F2) = SChain( tuple(F1.F..., F2) )
chain(F1, F2::SChain) = SChain( tuple(F1, F2.F...) )
chain(F1::SChain, F2::SChain) = chain( tuple(F1.F..., F2.F...) )

@generated function evaluate(chain::SChain{TC}, X)
   LEN = length(chain)  # inferred 
   code = Expr[]  
   push(code, :(X_0 = X))
   for l = 1:LEN 
      push!(code, Meta.parse("F_$l = chain.F[$l]")
      push!(code, Meta.parse("X_$l = evaluate(F_$l, X_$(l-1));")
   end
   append!(code, "return X_$LEN")
   Meta.parse(code)
end





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

