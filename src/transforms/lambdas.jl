
import ForwardDiff
import LegibleLambdas
import LegibleLambdas: @λ, LegibleLambda
import ACE: read_dict, write_dict 

import Serialization: serialize, deserialize


export @λ



evaluate(t::LegibleLambda, x) = t.λ(x)

evaluate_d(t::LegibleLambda, x::SVector) =  ForwardDiff.gradient(t.λ, x)

evaluate_dd(t::LegibleLambda, x::SVector) = ForwardDiff.hessian(t.λ, x)

evaluate_d(t::LegibleLambda, x::Real) =     ForwardDiff.derivative(t.λ, x)

evaluate_dd(t::LegibleLambda, x::Real) =     ForwardDiff.derivative(y -> evaluate_d(t, y), x)


function write_dict(t::LegibleLambda) 
   buf = IOBuffer()
   show(buf, t)
   return Dict(
         "__id__" => "ACE_LegibleLambda", 
         "ex" => "$(t.ex)",             # f is reconstructed from ex 
         "meta" => String(take!(buf))   # this will be ignored 
      )
end

function read_dict(::Val{:ACE_LegibleLambda}, D::Dict) 
   ex = Meta.parse(D["ex"])
   return LegibleLambda(ex, eval(ex))
end
