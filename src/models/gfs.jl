

@doc raw"""
```math
\varphi(R_i) = F(\varphi_1, \dots, \varphi_P)
```
Each ``\varphi_p`` is a linear ACE.

say more?
"""
struct GFinnisSinclairACE{TB, T, TEV, TF}
   basis::TB
   c::Matrix{T}
   evaluator::TEV
   F::TF
end

#test

# * constructors: 
#    basis, F, maybe c -> GFinnisSinclairACE
# * parameter wrangling: 
#      nparams, params, set_params!
# * evaluation codes: start with 
#      - evaluate 

# design interface for F, e.g., 
#        evaluate(F, phi)
#        grad_config(F, phi)
#        grad_params(F, phi)
