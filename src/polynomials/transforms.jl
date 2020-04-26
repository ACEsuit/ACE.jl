
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module Transforms

import Base:   ==
import JuLIP:  cutoff
import JuLIP.FIO: read_dict, write_dict

abstract type DistanceTransform end

export PolyTransform, IdTransform


poly_trans(p, r0, r) = @fastmath(((1+r0)/(1+r))^p)

poly_trans_d(p, r0, r) = @fastmath((-p/(1+r0)) * ((1+r0)/(1+r))^(p+1))

poly_trans_inv(p, r0, x) = ( (1+r0)/(x^(1/p)) - 1 )


"""
Implements the distance transform
```
r -> ( (1+r0)/(1+r))^p
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
   Dict("__id__" => "SHIPs_PolyTransform", "p" => T.p, "r0" => T.r0)

PolyTransform(D::Dict) = PolyTransform(D["p"], D["r0"])

read_dict(::Val{:SHIPs_PolyTransform}, D::Dict) = PolyTransform(D)

transform(t::PolyTransform, r::Number) = poly_trans(t.p, t.r0, r)

transform_d(t::PolyTransform, r::Number) = poly_trans_d(t.p, t.r0, r)

inv_transform(t::PolyTransform, x::Number) = poly_trans_inv(t.p, t.r0, x)

(t::PolyTransform)(x) = transform(t, x)

"""
`IdTransform`: Implements the distance transform `z -> z`;
Primarily used for the z-coordinate for the EnvPairPots

Constructor: `IdTransform()`
"""
struct IdTransform <: DistanceTransform
end

write_dict(T::IdTransform) =  Dict("__id__" => "SHIPs_IdTransform")
IdTransform(D::Dict) = IdTransform()
read_dict(::Val{:SHIPs_IdTransform}, D::Dict) = IdTransform(D)
transform(t::IdTransform, z::Number) = z
transform_d(t::IdTransform, r::Number) = one(r)
inv_transform(t::IdTransform, x::Number) = x


end
