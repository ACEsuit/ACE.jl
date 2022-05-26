


import Base:   ==
import ACE: read_dict, write_dict, 
       transform, transform_d, transform_dd, inv_transform
       
export polytransform, morsetransform, agnesitransform

abstract type DistanceTransform end        

# ----- new transforms implementation 
import ACE: λ 

@deprecate PolyTransform(p, r0) polytransform(p, r0)
polytransform(p, r0) = λ("r -> ((1+$r0)/(1+r))^$p")

@deprecate IdTransform() idtransform()
idtransform() = λ("r -> r")

@deprecate MorseTransform(lambda, r0) morsetransform(lambda, r0)
morsetransform(lambda, r0) = λ("r -> exp(- $lambda * (r / $r0 - 1))")

@deprecate AgnesiTransform(args...) agnesitransform(args...)
agnesitransform(r0, p=2, a=(p-1)/(p+1)) = λ("r -> 1 / (1 + $a * (r / $r0)^$p)")


# ------------------------------------------------------
#  implementation of inverse transform 

import Roots: find_zero
function inv_transform(trans, x)
   # solve trans(r) = x
   r = find_zero(r -> trans(r) - x, 1.0) 
   if !(trans(r) ≈ x)
      @warn("inv_transform via find_zero didn't find a good solution")
   end
   return r 
end 


# ------------------------------------------------------
# generic ad codes for distance transforms 

import ACE: rrule_evaluate, frule_evaluate, evaluate 

evaluate(trans::DistanceTransform, r::Number) = transform(trans, r)

function frule_evaluate(trans::DistanceTransform, r::Number, ∇r)
   # ∇r will be an abstract array or an abstract array encapsulated in a state
   dt = transform_d(trans, r)
   return dt * ∇r
end

function rrule_evaluate(trans::DistanceTransform, r::Number, w::AbstractVector)
   # here, the typical scenario is that evaluate -> B(T(r)), so the rrule 
   # should give ∑_n u_n d/dr B_n(T(r)) = ∑_n u_n B_n'(T(r)) * T'(r)
   # i.e. w = ∑_n u_n B_n'(T(r)) which is scalar-like. 
   return transform_d(trans, r) * w
end

