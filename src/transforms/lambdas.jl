
import ForwardDiff
import LegibleLambdas
import LegibleLambdas: @λ, LegibleLambda
import ACE: read_dict, write_dict 


λ(str::String) = λ(Meta.parse(str))

λ(ex::Expr) = LegibleLambda(ex, eval(ex))

analytic(str::String) = legiblelambda(str)

evaluate(t::LegibleLambda, x) = t.λ(x)

evaluate_d(t::LegibleLambda, x::SVector) =  ForwardDiff.gradient(t.λ, x)

ACE.evaluate_ed(t::LegibleLambda, x) =  evaluate(t, x), evaluate_d(t, x)

evaluate_dd(t::LegibleLambda, x::SVector) = ForwardDiff.hessian(t.λ, x)

evaluate_d(t::LegibleLambda, x::Real) =     ForwardDiff.derivative(t.λ, x)

evaluate_dd(t::LegibleLambda, x::Real) =     ForwardDiff.derivative(y -> evaluate_d(t, y), x)


function frule_evaluate(t::LegibleLambda, x::Real, dx::SVector)
      f = evaluate(t, x)
      df = evaluate_d(t, x)
      return f, df * dx
end



function write_dict(t::LegibleLambda) 
   buf = IOBuffer()
   show(buf, t)
   return Dict(
         "__id__" => "ACE_LegibleLambda", 
         "ex" => "$(t.ex)",             # f is reconstructed from ex 
         "meta" => String(take!(buf))   # this will be ignored 
      )
end

read_dict(::Val{:ACE_LegibleLambda}, D::Dict) = λ(D["ex"])

import Base: ==

==(F1::LegibleLambda, F2::LegibleLambda) = (F1.ex == F2.ex)

