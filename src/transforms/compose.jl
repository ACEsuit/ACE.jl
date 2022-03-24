
# early ideas towards a composition, or chain implementation. 
# not yet needed, but will explore further. 

import ACE: evaluate 
import Base: ∘

struct Compose{TF1, TF2}
   F1::TF1 
   F2::TF2
end

∘(F1::DistanceTransform, F2::StateTransform) = Compose(F1, F2)

evaluate(f::Compose, x) = evaluate( F1, evaluate(F2, x) )

