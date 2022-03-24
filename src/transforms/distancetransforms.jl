


import Base:   ==
import ACE: read_dict, write_dict, 
       transform, transform_d, transform_dd, inv_transform

"""
`DistanceTransform`  - abstract supertype for transformations from real 
numbers to real numbers. Historically they are called DistanceTransform, 
but the inputs need not be positive i.e. need not be distances.
"""
abstract type DistanceTransform end

export PolyTransform, IdTransform, MorseTransform, AgnesiTransform

using ForwardDiff: derivative 

poly_trans(p, r0, r) = ((1+r0)/(1+r))^p

poly_trans_d(p, r0, r) = (-p/(1+r0)) * ((1+r0)/(1+r))^(p+1)

poly_trans_dd(p, r0, r) = derivative(r -> poly_trans_d(p, r0, r), r)

poly_trans_inv(p, r0, x) = ( (1+r0)/(x^(1/p)) - 1 )


# TODO: generalise the distance transform to allow
#       ((a + r0) / (a + r) )^p

@doc raw"""
Implements the distance transform
```math
   x(r) = \Big(\frac{1 + r_0}{1 + r}\Big)^p
```

Constructor:
```
PolyTransform(p, r0)
```
"""
struct PolyTransform{TP, T} <: DistanceTransform
   p::TP
   r0::T
end

write_dict(T::PolyTransform) =
   Dict("__id__" => "ACE_PolyTransform", "p" => T.p, "r0" => T.r0)

PolyTransform(D::Dict) = PolyTransform(D["p"], D["r0"])

read_dict(::Val{:ACE_PolyTransform}, D::Dict) = PolyTransform(D)

transform(t::PolyTransform, r::Number) = poly_trans(t.p, t.r0, r)

transform_d(t::PolyTransform, r::Number) = poly_trans_d(t.p, t.r0, r)

transform_dd(t::PolyTransform, r::Number) = poly_trans_dd(t.p, t.r0, r)

inv_transform(t::PolyTransform, x::Number) = poly_trans_inv(t.p, t.r0, x)

(t::PolyTransform)(x) = transform(t, x)

"""
`IdTransform`: Implements the distance transform `z -> z`;
Primarily used for the z-coordinate for the EnvPairPots

Constructor: `IdTransform()`
"""
struct IdTransform <: DistanceTransform
end

write_dict(T::IdTransform) =  Dict("__id__" => "ACE_IdTransform")
IdTransform(D::Dict) = IdTransform()
read_dict(::Val{:ACE_IdTransform}, D::Dict) = IdTransform(D)
transform(t::IdTransform, z::Number) = z
transform_d(t::IdTransform, r::Number) = one(r)
transform_dd(t::IdTransform, r::Number) = zero(r)
inv_transform(t::IdTransform, x::Number) = x

read_dict(::Val{:SHIPs_IdTransform}, D::Dict) =
   read_dict(Val{:ACE_IdTransform}(), D)


@doc raw"""
Implements the distance transform
```math
   x(r) = \exp( - \lambda (r/r_0) )
```

Constructor:
```
MorseTransform(lambda, r0)
```
"""
struct MorseTransform{T} <: DistanceTransform
   lambda::T
   r0::T
end

write_dict(T::MorseTransform) =
   Dict("__id__" => "ACE_MorseTransform", "lambda" => T.p, "r0" => T.r0)
MorseTransform(D::Dict) = MorseTransform(D["lambda"], D["r0"])
read_dict(::Val{:ACE_MorseTransform}, D::Dict) = MorseTransform(D)
transform(t::MorseTransform, r::Number) = exp(- t.lambda * (r/t.r0))
transform_d(t::MorseTransform, r::Number) = (-t.lambda/t.r0) * exp(- t.lambda * (r/t.r0))
inv_transform(t::MorseTransform, x::Number) = - t.r0/t.lambda * log(x)
(t::MorseTransform)(x) = transform(t, x)


@doc raw"""
Implements the distance transform
```math
   x(r) = \frac{1}{1 + a (r/r_0)^p}
```
with default $a = (p-1)/(p+1)$. That default is chosen such that
$|x'(r)|$ is maximised at $r = r_0$. Default for $p$ is $p = 2$. Any value
$p > 1$ is permitted.

Constructor:
```
AgnesiTransform(r0, [p, [, a]])
```
"""
struct AgnesiTransform{T, TP} <: DistanceTransform
   r0::T
   p::TP
   a::T
   c::T
end

AgnesiTransform(r0, p=2, a=(p-1)/(p+1)) = (@assert p > 1;
                             AgnesiTransform(r0, p, a, - a * p / r0))

write_dict(T::AgnesiTransform) =
Dict("__id__" => "ACE_AgnesiTransform", "r0" => T.r0, "p" => T.p, "a" => T.a)
AgnesiTransform(D::Dict) = AgnesiTransform(D["r0"], D["p"], D["a"])
read_dict(::Val{:ACE_AgnesiTransform}, D::Dict) = AgnesiTransform(D)
transform(t::AgnesiTransform, r::Number) = @fastmath 1 / (1 + t.a * (r/t.r0)^t.p)
transform_d(t::AgnesiTransform, r::Number) = (s1 = (r/t.r0); s2 = s1^(t.p-1);
                                         @fastmath t.c * s2 / (1+t.a * s2*s1)^2)
inv_transform(t::AgnesiTransform, x::Number) = t.r0 * ( (1/x-1)/t.a )^(1/t.p)
(t::AgnesiTransform)(x) = transform(t, x)


