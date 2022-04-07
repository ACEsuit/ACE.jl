

"""
Types derived from `B1pMultiplier` can be used to multiply a 1-p basis 
with another function. They are implemented as another 1-p basis, but with just 
a single basis function attached to them. In particular they don't get 
a basis function index attached to them. The abstract type 
`abstract type B1pMultiplier` implements the generic functionality 
to enable this. 

Derived types must implement 
* `_inner_evaluate`
* `_inner_evaluate_d`
"""
abstract type B1pMultiplier{T} <: OneParticleBasis{T} end 

valtype(::B1pMultiplier{T}, args...)  where {T} = T 

function _inner_evaluate end 
function _inner_evaluate_d end 

function evaluate!(A, mult::B1pMultiplier, X::AbstractState) 
    A[1] = _inner_evaluate(mult, X)
    return A 
end 

symbols(::B1pMultiplier) = Tuple{}()

indexrange(::B1pMultiplier) = Dict() 

Base.length(basis::B1pMultiplier) = 1

isadmissible(b, ::B1pMultiplier) = true 

degree(b, ::B1pMultiplier, args...) = 0

get_index(::B1pMultiplier, b) = 1

get_spec(::B1pMultiplier, n::Integer) = NamedTuple()




