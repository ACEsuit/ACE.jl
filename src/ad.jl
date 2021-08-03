
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

