
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



module Transforms

import Base:   ==
import JuLIP:  cutoff, AtomicNumber
import JuLIP.FIO: read_dict, write_dict

abstract type DistanceTransform end

export PolyTransform, IdTransform, MorseTransform, AgnesiTransform

# fall-backs from species dependent to independent transforms 
transform(t::DistanceTransform, r::Number, z::AtomicNumber, z0::AtomicNumber) = transform(t, r)
transform_d(t::DistanceTransform, r::Number, z::AtomicNumber, z0::AtomicNumber) = transform_d(t, r)
inv_transform(t::DistanceTransform, x::Number, z::AtomicNumber, z0::AtomicNumber) = inv_transform(t, x)



poly_trans(p, r0, r) = @fastmath(((1+r0)/(1+r))^p)

poly_trans_d(p, r0, r) = @fastmath((-p/(1+r0)) * ((1+r0)/(1+r))^(p+1))

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

read_dict(::Val{:SHIPs_PolyTransform}, D::Dict) =
   read_dict(Val{:ACE_PolyTransform}(), D::Dict)

read_dict(::Val{:ACE_PolyTransform}, D::Dict) = PolyTransform(D)

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

write_dict(T::IdTransform) =  Dict("__id__" => "ACE_IdTransform")
IdTransform(D::Dict) = IdTransform()
read_dict(::Val{:ACE_IdTransform}, D::Dict) = IdTransform(D)
transform(t::IdTransform, z::Number) = z
transform_d(t::IdTransform, r::Number) = one(r)
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


# --------- Utility function - affine transform 

"""
`AffineT` : wraps another transform and then applies an affine transformation. 

Constructor: 
```julia
AffineT(transform, x1, x2, y1, y2)
```
then `x = t(r)` and then `x -> y` with `xi -> yi`.
"""
struct AffineT{T, TT} <: DistanceTransform
   t::TT   # the inner transform  : t(r) = x
   x1::T   # intervals to be transformed x1 -> y1 etc...
   x2::T 
   y1::T 
   y2::T
end

transform(t::AffineT, r) = t.y1 + (transform(t.t, r) - t.x1) * (t.y2-t.y1)/(t.x2-t.x1)
transform_d(t::AffineT, r) = ((t.y2-t.y1)/(t.x2-t.x1)) * transform_d(t.t, r)
transform_inv(t::AffineT, y) = transform_inv(t.t, t.x1 + (y - t.y1) * (t.x2-t.x1)/(t.y2-t.y1))

write_dict(T::AffineT) = 
      Dict("__id__" => "ACE_AffineT", 
           "t" => write_dict(T.t), 
           "xy" => [T.x1, T.x2, T.y1, T.y2] )

read_dict(::Val{:ACE_AffineT}, D::Dict) = AffineT(D) 

AffineT(D::Dict) = AffineT(read_dict(D["t"]), 
                         D["xy"]... )

# --------- Multi-transform: species-dependent transform 

import JuLIP: chemical_symbol
import JuLIP.Potentials: ZList, SZList, i2z, z2i
using StaticArrays: SMatrix
struct MultiTransform{NZ, TT} <: DistanceTransform
   zlist::SZList{NZ}
   transforms::SMatrix{NZ, NZ, TT}
end 

# FIO 

write_dict(T::MultiTransform) =
      Dict("__id__" => "ACE_MultiTransform", 
           "zlist" => write_dict(T.zlist), 
           "transforms" => write_dict.(T.transforms[:]))

read_dict(::Val{:ACE_MultiTransform}, D::Dict) = MultiTransform(D)

function MultiTransform(D::Dict) 
   zlist = read_dict(D["zlist"])
   NZ = length(zlist) 
   transforms = SMatrix{NZ, NZ}( read_dict.(D["transforms"])... )
   return MultiTransform(zlist, transforms)   
end

#  Constructor 

function multitransform(D::Dict; rin=nothing, rcut=nothing)
   species = Symbol[] 
   for key in keys(D) 
      append!(species, [key...])
   end
   species = unique(species) 
   zlist = ZList(species, static=true)
   NZ = length(zlist) 
   transforms = Matrix{Any}(undef, NZ, NZ) 
   for i = 1:NZ, j = 1:NZ 
      Si = chemical_symbol(i2z(zlist, i))
      Sj = chemical_symbol(i2z(zlist, j))
      if haskey(D, (Si, Sj))
         t = D[(Si, Sj)]
      else
         t = D[(Sj, Si)]
      end
      # apply an affine transform so all transforms have the same range 
      if rin != nothing && rcut != nothing 
         x1 = transform(t, rin)
         x2 = transform(t, rcut)
         transforms[i, j] = AffineT(t, x1, x2, -1.0, 1.0)
      else 
         transforms[i, j] = t
      end
   end
   transforms = identity.(transforms)  # infer the type 
   return MultiTransform(zlist, SMatrix{NZ, NZ}(transforms...))
end

transform(t::MultiTransform, r::Number, z::AtomicNumber, z0::AtomicNumber) = 
      transform(t.transforms[z2i(t.zlist, z), z2i(t.zlist, z0)], r)

transform_d(t::MultiTransform, r::Number, z::AtomicNumber, z0::AtomicNumber) =
      transform_d(t.transforms[z2i(t.zlist, z), z2i(t.zlist, z0)], r)

inv_transform(t::MultiTransform, x::Number, z::AtomicNumber, z0::AtomicNumber) =
      inv_transform(t.transforms[z2i(t.zlist, z), z2i(t.zlist, z0)], r)

# NOTE: This is a bit of a hack, I'm checking whether 
#       the transforms have been transformed to the domain [-1, 1]
#       then I'm checking whether r is either rin or rcut and only 
#       then do I return the value
function transform(t::MultiTransform{NZ, TT}, r::Number) where {NZ, TT}
   @assert (TT <: AffineT) "transform(::MultiTransfrom, r) is only defined if rin, rcut are specified during construction"
   x = sum( transform(_t, r)  for _t in t.transforms ) / NZ^2 
   @assert (abs(abs(x) - 1) <= 1e-7) "transform(::MultiTransfrom, r) is only defined for r = rin, rcut"
   return x  
end

end 
