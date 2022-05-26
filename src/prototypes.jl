
export SimpleSparseBasis, SparseBasis, MaxBasis 



# prototypes for space transforms and cutoffs
function transform end
function transform_d end
function transform_dd end
function inv_transform end


# ------------------------------------------------------------
#  Abstract polynomial degree business


"""
interface functions for `OneParticleBasis`
"""
function add_into_A! end

"""
interface functions for `OneParticleBasis`
"""
function add_into_A_dA! end


"""
`function degree` : compute some notion of degree of
the `arg` argument.
"""
function degree end

"""
`function scaling(b, p)`:

a scaling factor for a basis functions ϕ, which gives a rought estimate on
   the magnitude of ∇ᵖϕ e.g.,
```
ϕ = r^n Ylm
```
has scaling factor `n^p + l^p`, though sharper estimates are also possible.
"""
function scaling end


using LinearAlgebra: Diagonal

diagonal_regulariser(basis; diff = 0) = Diagonal(scaling(basis, diff))

"""
every scalar basis must implement this
"""
function rand_radial end

@doc raw"""
```math
\begin{aligned}
a &= A(x)
b &= B(a) = B(A(x))
∂b/∂x &= ∂b/∂a ⋅ ∂a/∂x
\end{aligned}
```
In code this becomes 
```julia
a = evaluate(A, x)
b = evaluate(B, a)
∂a∂x = evaluate_d(A, x)
∂b∂x = frule_evaluate(B, a, ∂a∂x)
```
"""
function frule_evaluate! end 


@doc raw"""
```math
\begin{aligned}
a &= A(x)
b &= B(a) = B(A(x))
∂b/∂x &= ∂b/∂a ⋅ ∂a/∂x
\end{aligned}
```
In code this becomes 
```julia
a = evaluate(A, x)
b = evaluate(B, a)
∂b∂a = evaluate_d(B, a)
∂b∂x = rrule_evaluate(∂b∂a, A, x)
```
"""
function rrule_evaluate! end 
