
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# This file implements different ways to specify a degree

using JuLIP.Chemistry: atomic_number

import Base: ==

export SparseSHIPBasis, HyperbolicCross



abstract type BasisSpec{BO, NZ} end
abstract type AnalyticBasisSpec{BO, NZ} <: BasisSpec{BO, NZ} end

"""
`get_basisspec(<: BasisSpec) -> allKL, Nu`
"""
function get_basisspec end


VecOrTup = Union{AbstractVector, Tuple}

_convert_Zs(Zs::Symbol) = (atomic_number(Zs),)
_convert_Zs(Zs::Integer) = (Zs,)
_convert_Zs(Zs::VecOrTup) = _convert_Zs( tuple(Zs...) )
_convert_Zs(Zs::NTuple{NZ, Symbol}) where {NZ} = atomic_number.(Zs)
_convert_Zs(Zs::NTuple{NZ, <: Integer}) where {NZ} = Int16.(Zs)



"""
`SparseSHIPBasis` : a sparse-grid type degree definition,
```
deg({k}, {l}) = ∑ (k + wL * l)
```
"""
struct SparseSHIPBasis{BO, NZ} <: AnalyticBasisSpec{BO, NZ}
   deg::IntS
   wL::Float64
   Zs::NTuple{NZ, Int16}
   valbo::Val{BO}
   z2i::Dict{Int16, Int16}
end

==(s1::SparseSHIPBasis, s2::SparseSHIPBasis) =
      all( getfield(s1, i) == getfield(s2, i)
           for i = 1:fieldcount(SparseSHIPBasis) )

SparseSHIPBasis(bo::Integer, deg::Integer, wL::Real) =
      SparseSHIPBasis(bo::Integer, :X, deg::Integer, wL::Real)

function SparseSHIPBasis(bo::Integer, Zs, deg::Integer, wL::Real)
   @assert wL > 0
   @assert deg > 0
   @assert bo >= 0
   Zs = _convert_Zs(Zs)
   z2i = Dict([ Int16(z) => Int16(i) for (i, z) in enumerate(Zs) ]...)
   return SparseSHIPBasis(IntS(deg), Float64(wL), Zs, Val(bo), z2i)
end

z2i(spec::SparseSHIPBasis, z) = spec.z2i[z]
i2z(spec::SparseSHIPBasis, z) = spec.Zs[i]

numspecies(spec::SparseSHIPBasis) = length(spec.Zs) 

deg(D::SparseSHIPBasis, k::Integer, l::Integer) =
      k + D.wL * l

deg(D::SparseSHIPBasis, kk::VecOrTup, ll::VecOrTup) =
      sum( deg(D, k, l) for (k, l) in zip(kk, ll) )

admissible(D::SparseSHIPBasis, k, l) = deg(D, k, l) <= D.deg

maxK(D::SparseSHIPBasis) = D.deg

# For a pure 2-body potential we don't need an angular component
maxL(D::SparseSHIPBasis{1}, args...) = 0

maxL(D::SparseSHIPBasis) = floor(Int, D.deg / D.wL)

maxL(D::SparseSHIPBasis, k::Integer) = floor(Int, (D.deg - k) / D.wL)

Dict(D::SparseSHIPBasis{BO}) where {BO} = Dict("__id__" => "SHIPs_SparseSHIPBasis",
                                "deg" => D.deg,
                                "wL" => D.wL,
                                "Zs" => D.Zs,
                                "bo" => BO)

convert(::Val{:SHIPs_SparseSHIPBasis}, D::Dict) =
      SparseSHIPBasis(D["bo"], D["Zs"], D["deg"], D["wL"])


# """
# `HyperbolicCross` : standard hyperbolic cross degree,
# ```
# deg({k}, {l}) = prod( max(1, k + wL * l) )
# ```
# """
# struct HyperbolicCross <: BasisSpec
#    deg::Int
#    wL::Float64
# end
#
# deg(D::HyperbolicCross, kk::VecOrTup, ll::VecOrTup) =
#       prod( max(1, deg(D, k, l)) for (k, l) in zip(kk, ll) )
# maxK(D::HyperbolicCross) = D.deg
# maxL(D::HyperbolicCross) = floor(Int, D.deg / D.wL)
# maxL(D::HyperbolicCross, k::Integer) = floor(Int, (D.deg - k) / D.wL)
#
# Dict(D::HyperbolicCross) = Dict("__id__" => "SHIPs_HyperbolicCross",
#                             "deg" => D.deg, "wL" => D.wL)
# convert(::Val{:SHIPs_HyperbolicCross}, D::Dict) =
#       HyperbolicCross(D["deg"], D["wL"])
#



function generate_KL(spec::AnalyticBasisSpec, TI = IntS, TF=Float64)
   allKL = NamedTuple{(:k, :l), Tuple{TI,TI}}[]
   degs = TF[]
   # find k, l such that deg(spec, k, l) ≦ deg
   for k = 0:maxK(spec), l = 0:maxL(spec, k)
      @assert admissible(spec, k, l)
      push!(allKL, (k=k, l=l))
      push!(degs, deg(spec, k, l))
   end
   # sort allKL according to total degree
   I = sortperm(degs)
   return allKL[I], degs[I]
end

# NOTE: this is a very rudimentary generate_ZKL that could be
#       overloaded and so that different `allKL` collections are
#       created for each species.

function generate_ZKL(z2i, spec::BasisSpec, TI = IntS, TF=Float64)
   allKL, degs = generate_KL(spec, TI, TF)
   allZKL = [ allKL for _=1:length(z2i) ]
   return allZKL
end


"""
return kk, ll, mrange
where kk, ll is BO-tuples of k and l indices, while mrange is a
cartesian range over which to iterate to construct the basis functions

(note: this is tested for correcteness and speed)
"""
function _klm(ν::StaticVector{BO}, KL) where {BO}
   kk = SVector( ntuple(i -> KL[ν[i]].k, BO) )
   ll = SVector( ntuple(i -> KL[ν[i]].l, BO) )
   return kk, ll, _mrange(ll)
end


"""
create a vector of Nu arrays with the right type information
for each body-order
"""
function _init_Nu(bo::Integer, TI=IntS)
   Nu = []
   for n = 1:bo
      push!(Nu, SVector{n, TI}[])
   end
   # convert into an SVector to make the length a type parameters
   return tuple(Nu...)
end

function generate_KL_tuples(spec::AnalyticBasisSpec{BO}, cg;
                            filter=true) where {BO}
   # all possible (k, l) pairs
   allKL, degs = generate_KL(spec)
   # sepatare arrays for all body-orders
   Nu = _init_Nu(BO)
   for N = 1:BO
      _generate_KL_tuples!(Nu[N], spec, cg, allKL, degs; filter=filter)
   end
   return allKL, Nu
end

function _generate_KL_tuples!(Nu::Vector{<: SVector{BO}}, Deg::BasisSpec,
                             cg, allKL, degs;
                             filter=true) where {BO}
   # the first iterm is just (0, ..., 0)
   # we can choose (k1, l1), (k2, l2) ... by indexing into allKL
   # then we start incrementing until we hit the maximum degree
   # while retaining the ordering ν₁ ≤ ν₂ ≤ …
   lastidx = 0
   ν = @MVector ones(IntS, BO)   # (ones(IntS, bo)...)
   while true
      # check whether the current ν tuple is admissible
      # the first condition is that its max index is small enough
      isadmissible = (maximum(ν) <= length(allKL))
      if isadmissible
         # the second condition is that the multivariate degree it defines
         # is small enough => for that we first have to compute the corresponding
         # k and l vectors
         kk, ll, _ = _klm(ν, allKL)
         isadmissible = admissible(Deg, kk, ll)
      end

      # we want to increment `curindex`, but if we've reach the maximum degree
      # then we need to move to the next index down

      # if the current tuple ν has admissible degree ...
      if isadmissible
         # ... then we add it to the stack  ...
         #     (at least if it is an admissible basis function respecting
         #      all the symmetries - this is checked by filter_tuple)
         if !filter || filter_tuple(allKL, ν, cg)
            push!(Nu, SVector(ν))
         end
         # ... and increment it
         lastidx = BO
         ν[lastidx] += 1
      else
         # we have overshot, _deg(ν) > deg; we must go back down, by
         # decreasing the index at which we increment
         if lastidx == 1
            break
         end
         ν[lastidx-1:end] .= ν[lastidx-1] + 1
         lastidx -= 1
      end
   end
   return allKL, Nu
end
