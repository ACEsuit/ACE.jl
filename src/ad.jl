
# Experimental AD codes

import ChainRulesCore, ChainRules
import ChainRulesCore: rrule, NO_FIELDS 

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
   adj = dp -> ( NO_FIELDS, dp * getproperty.(grad_params(y.m, y.X), :val)) 
   return val, adj
end


function (y::EVAL_D)(params)
   set_params!(y.m, params)
   return grad_config(y.m, y.X)
end


function rrule(y::EVAL_D, params)
   set_params!(y.m, params)
   val = grad_config(y.m, y.X)
   adj = dp -> ( NO_FIELDS, 
         begin 
            dB = grad_params_config(y.m, y.X)
            g = zeros(size(dB, 1))
            for i = 1:length(g), j = 1:size(dB, 2)
               g[i] += dot(dB[i, j], dp[j])
            end
            g
         end
      )
   return val, adj
end

