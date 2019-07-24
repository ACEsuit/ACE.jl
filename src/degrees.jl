
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# This file implements different ways to specify a degree


export TotalDegree, HyperbolicCross

abstract type AbstractDegree end

VecOrTup = Union{AbstractVector, Tuple}

admissible(D::AbstractDegree, k, l) = deg(D, k, l) <= D.deg

deg(D::AbstractDegree, k::Integer, l::Integer) =
      k + D.wL * l

"""
`TotalDegree` : a sparse-grid type degree definition,
```
deg({k}, {l}) = ∑ (k + wL * l)
```
"""
struct TotalDegree <: AbstractDegree
   deg::Int
   wL::Float64
end

deg(D::TotalDegree, kk::VecOrTup, ll::VecOrTup) =
      sum( deg(D, k, l) for (k, l) in zip(kk, ll) )
maxK(D::TotalDegree) = D.deg
maxL(D::TotalDegree) = floor(Int, D.deg / D.wL)
maxL(D::TotalDegree, k::Integer) = floor(Int, (D.deg - k) / D.wL)

Dict(D::TotalDegree) = Dict("__id__" => "SHIPs_TotalDegree",
                            "deg" => D.deg, "wL" => D.wL)
convert(::Val{:SHIPs_TotalDegree}, D::Dict) =
      TotalDegree(D["deg"], D["wL"])


"""
`HyperbolicCross` : standard hyperbolic cross degree,
```
deg({k}, {l}) = prod( max(1, k + wL * l) )
```
"""
struct HyperbolicCross <: AbstractDegree
   deg::Int
   wL::Float64
end

deg(D::HyperbolicCross, kk::VecOrTup, ll::VecOrTup) =
      prod( max(1, deg(D, k, l)) for (k, l) in zip(kk, ll) )
maxK(D::HyperbolicCross) = D.deg
maxL(D::HyperbolicCross) = floor(Int, D.deg / D.wL)
maxL(D::HyperbolicCross, k::Integer) = floor(Int, (D.deg - k) / D.wL)

Dict(D::HyperbolicCross) = Dict("__id__" => "SHIPs_HyperbolicCross",
                            "deg" => D.deg, "wL" => D.wL)
convert(::Val{:SHIPs_HyperbolicCross}, D::Dict) =
      HyperbolicCross(D["deg"], D["wL"])




function generate_KL(D::AbstractDegree, TI = IntS, TF=Float64)
   allKL = NamedTuple{(:k, :l, :deg), Tuple{TI,TI,TF}}[]
   degs = TF[]
   # morally "k + wL * l <= deg"
   for k = 0:maxK(D), l = 0:maxL(D, k)
      push!(allKL, (k=k, l=l, deg=deg(D, k, l)))
      push!(degs, deg(D, k, l))
   end
   # sort allKL according to total degree
   I = sortperm(degs)
   return allKL[I], degs[I]
end

# NOTE: this is a very rudimentary generate_ZKL that could be
#       overloaded and so that different `allKL` collections are
#       created for each species.

function generate_ZKL(z2i, D::AbstractDegree, TI = IntS, TF=Float64)
   allKL, degs = generate_KL(D, TI, TF)
   allZKL = [ allKL for _=1:length(z2i) ]
   return allZKL
end
