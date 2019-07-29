
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# This file implements different ways to specify a degree

using Combinatorics
using JuLIP.Chemistry: atomic_number

import Base: ==

export SparseSHIP, HyperbolicCrossSHIP



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


z2i(spec::AnalyticBasisSpec, z) = spec.z2i[z]

i2z(spec::AnalyticBasisSpec, z) = spec.Zs[i]

nspecies(spec::AnalyticBasisSpec) = length(spec.Zs)

admissible(D::AnalyticBasisSpec, k, l) = deg(D, k, l) <= D.deg



"""
`SparseSHIP` : a sparse-grid type degree definition,
```
deg({k}, {l}) = ∑ (k + wL * l)
```
"""
struct SparseSHIP{BO, NZ} <: AnalyticBasisSpec{BO, NZ}
   deg::IntS
   wL::Float64
   Zs::NTuple{NZ, Int16}
   valbo::Val{BO}
   z2i::Dict{Int16, Int16}
end

==(s1::SparseSHIP, s2::SparseSHIP) =
      all( getfield(s1, i) == getfield(s2, i)
           for i = 1:fieldcount(SparseSHIP) )


SparseSHIP(bo::Integer, deg::Integer, wL::Real) =
      SparseSHIP(bo::Integer, :X, deg::Integer, wL::Real)

function SparseSHIP(bo::Integer, Zs, deg::Integer, wL::Real)
   @assert wL > 0
   @assert deg > 0
   @assert bo >= 0
   Zs = _convert_Zs(Zs)
   z2i = Dict([ Int16(z) => Int16(i) for (i, z) in enumerate(Zs) ]...)
   return SparseSHIP(IntS(deg), Float64(wL), Zs, Val(bo), z2i)
end


deg(D::SparseSHIP, k::Integer, l::Integer) =
      k + D.wL * l

deg(D::SparseSHIP, kk::VecOrTup, ll::VecOrTup) =
      sum( deg(D, k, l) for (k, l) in zip(kk, ll) )

maxK(D::SparseSHIP) = D.deg

# For a pure 2-body potential we don't need an angular component
maxL(D::SparseSHIP{1}, args...) = 0

maxL(D::SparseSHIP) = floor(Int, D.deg / D.wL)

maxL(D::SparseSHIP, k::Integer) = floor(Int, (D.deg - k) / D.wL)

Dict(D::SparseSHIP{BO}) where {BO} = Dict("__id__" => "SHIPs_SparseSHIP",
                                "deg" => D.deg,
                                "wL" => D.wL,
                                "Zs" => D.Zs,
                                "bo" => BO)

convert(::Val{:SHIPs_SparseSHIP}, D::Dict) =
      SparseSHIP(D["bo"], D["Zs"], D["deg"], D["wL"])

# ---------------------------------------------------------------


"""
`HyperbolicCrossSHIP` : standard hyperbolic cross degree,
```
deg({k}, {l}) = prod( max(a, b + k + wL * l) )
```
default is `a = 1, b = 0`
"""
struct HyperbolicCrossSHIP{BO, NZ} <: AnalyticBasisSpec{BO, NZ}
   deg::IntS
   wL::Float64
   a::Float64
   b::Float64
   Zs::NTuple{NZ, Int16}
   valbo::Val{BO}
   z2i::Dict{Int16, Int16}
end

==(s1::HyperbolicCrossSHIP, s2::HyperbolicCrossSHIP) =
      all( getfield(s1, i) == getfield(s2, i)
           for i = 1:fieldcount(HyperbolicCrossSHIP) )

HyperbolicCrossSHIP(bo::Integer, deg::Integer, wL::Real; kwargs...) =
      HyperbolicCrossSHIP(bo::Integer, :X, deg::Integer, wL::Real; kwargs...)

function HyperbolicCrossSHIP(bo::Integer, Zs, deg::Integer, wL::Real;
                             a = 1.0, b = 0.0)
   @assert wL > 0
   @assert deg > 0
   @assert bo >= 0
   Zs = _convert_Zs(Zs)
   z2i = Dict([ Int16(z) => Int16(i) for (i, z) in enumerate(Zs) ]...)
   return HyperbolicCrossSHIP(IntS(deg), Float64(wL), Float64(a), Float64(b),
                              Zs, Val(bo), z2i)
end


deg(spec::HyperbolicCrossSHIP, k::Integer, l::Integer) =
      k + spec.wL * l

deg(spec::HyperbolicCrossSHIP, kk::VecOrTup, ll::VecOrTup) =
      prod( max(spec.a, spec.b + deg(spec, k, l)) for (k, l) in zip(kk, ll) )

maxK(spec::HyperbolicCrossSHIP) = floor(Int, spec.deg - spec.b)

maxL(spec::HyperbolicCrossSHIP) = floor(Int, (spec.deg - spec.b) / spec.wL)

maxL(spec::HyperbolicCrossSHIP, k::Integer) = floor(Int, (spec.deg - spec.b - k) / spec.wL)

Dict(D::HyperbolicCrossSHIP) = Dict("__id__" => "SHIPs_HyperbolicCrossSHIP",
                            "deg" => D.deg, "wL" => D.wL)
convert(::Val{:SHIPs_HyperbolicCrossSHIP}, D::Dict) =
      HyperbolicCrossSHIP(D["deg"], D["wL"])



# ---------------------------------------------------------------


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

function generate_ZKL(spec::AnalyticBasisSpec, TI = IntS, TF=Float64)
   allKL, degs = generate_KL(spec, TI, TF)
   allZKL = ntuple( _->copy(allKL), nspecies(spec) )
   return return allZKL
end


"""
returns kk, ll
where kk, ll is BO-tuples of k and l indices
"""
function _kl(ν::StaticVector{N}, KL) where {N}
   kk = SVector( ntuple(i -> KL[i][ν[i]].k, N) )
   ll = SVector( ntuple(i -> KL[i][ν[i]].l, N) )
   return kk, ll
end


"""
create a vector of Nu arrays with the right type information
for each body-order
"""
function _init_Nu(bo::Integer, nz::Integer, TI=IntS)
   Nu = Matrix{Vector}(undef, bo, nz)
   for n = 1:bo, iz = 1:nz
      Nu[n, iz] = SVector{n, TI}[]
   end
   # convert into an SVector to make the length a type parameters
   return SMatrix{bo,nz}(Nu)
end

"""
create a vector of Nu arrays with the right type information
for each body-order
"""
function _init_NuZ(bo::Integer, nz::Integer, TI=IntS)
   Nu = Matrix{Vector}(undef, bo, nz)
   for n = 1:bo, iz = 1:nz
      Nu[n, iz] = NamedTuple{       (:izz,              :ν),
                               Tuple{SVector{n, Int16}, SVector{n, TI}} }[]
   end
   # convert into an SVector to make the length a type parameters
   return SMatrix{bo,nz}(Nu)
end


function generate_ZKL_tuples(spec::AnalyticBasisSpec{BO}, cg;
                            filter=true) where {BO}
   # all possible (k, l) pairs
   allZKL = generate_ZKL(spec)
   # separate arrays for all body-orders and species
   NuZ = _init_NuZ(BO, nspecies(spec))
   for N = 1:BO
      _generate_ZKL_tuples!(NuZ[N,1], spec, cg, allZKL, Val(N); filter=filter)
      for iz = 2:nspecies(spec)
         append!(NuZ[N, iz], NuZ[N, 1])
      end
   end
   return allZKL, NuZ
end


"""
takes two N-body basis functions each specified by a species vector `zzi` and
an index vector `νi` into the A-basis and decides whether they define the
same basis function. E.g.,
```
ν = SVector(1,2,2)
z1 = SVector(1,1,2)
z2 = SVector(1,2,1)
z3 = SVector(2,1,1)
@show _iseqB(z1, ν, z2, ν) # -> true
@show _iseqB(z1, ν, z3, ν) # -> false
```
"""
@generated function _iseqB(zz1::StaticVector{N}, ν1::StaticVector{N},
                           zz2::StaticVector{N}, ν2::StaticVector{N}
                           ) where {N}
   ex1 = "a1 = SVector(" * prod("(zz1[$i], ν1[$i])," for i = 1:N)[1:end-1] * ")"
   ex2 = "a2 = SVector(" * prod("(zz2[$i], ν2[$i])," for i = 1:N)[1:end-1] * ")"
   quote
      $(Meta.parse(ex1))
      $(Meta.parse(ex2))
      return sort(a1) == sort(a2)
   end
end


function _generate_ZKL_tuples!(NuZ, spec::AnalyticBasisSpec, cg, ZKL, ::Val{BO};
                               filter=true) where {BO}
   nz = nspecies(spec)
   izz = @MVector ones(Int16, BO)
   izz_tmp = SVector{BO, Int16}[]

   # temporary storage for all ν-tuples for a given zz combination
   Nu = SVector{BO, IntS}[]
   Nu_old = SVector{BO, IntS}[]

   for izz_ci in CartesianIndices(ntuple(_->(1:nz), BO))
      izz = Int16.(SVector(Tuple(izz_ci)...))
      # don't use this zz unless it is sorted; we will look at all permutations
      # of zz below; this means lots of skipping, but who cares, this part of
      # the loop is cheap. All the cost comes later.
      if !issorted(Tuple(izz))
         continue
      end

      # collect all possible ν tuples for the current species combination
      # specified by `zz`.  A (zz, ν) combination specifies a basis function
      #    ∏_a A[zₐ][νₐ]     (actually izₐ instead of zₐ and ignoring the m's)
      empty!(Nu)
      _generate_KL_tuples!(Nu, spec, cg, ZKL[izz]; filter=filter)

      # now loop through all the ν tuples we found to push them into NuZ
      for ν in Nu
         # for each such ν we also need to check whether any permutations
         # of `izz` give a new basis function as well. E.g.,
         #   A[1,n] A[1,m] == A[1,m] A[1,n]
         #   A[1,n] A[2,n] == A[2,n] A[1,n]
         # but
         #   A[1,n] A[2,m] != A[2,n] A[1,m]
         empty!(izz_tmp)
         push!(izz_tmp, izz)
         push!(NuZ, (izz = izz, ν = ν))
         for izzp in unique(permutations(izz))
            if !any( _iseqB(SVector(izzp...), ν, izz1, ν)  for izz1 in izz_tmp )
               push!(izz_tmp, izzp)
               push!(NuZ, (izz = izzp, ν = ν))
            end
         end
      end
   end
end


function _generate_KL_tuples!(Nu::Vector{<: SVector{BO}},
                              spec::AnalyticBasisSpec,
                              cg, ZKLs;
                              filter=true) where {BO}
   # the first item is just (1, ..., 1)
   # we can choose (k1, l1), (k2, l2) ... by indexing into allKL
   # then we start incrementing until we hit the maximum degree
   # while retaining the ordering ν₁ ≤ ν₂ ≤ …
   lastidx = 0
   ν = @MVector ones(IntS, BO)
   kk, ll = _kl(ν, ZKLs)
   while true
      # check whether the current ν tuple is admissible
      # the first condition is that its max index is small enough
      isadmissible = all(ν[i] <= length(ZKLs[i]) for i = 1:BO)
      if isadmissible
         # the second condition is that the multivariate degree it defines
         # is small enough => for that we first have to compute the corresponding
         # k and l vectors
         kk, ll = _kl(ν, ZKLs)
         isadmissible = admissible(spec, kk, ll)
      end

      # we want to increment `curindex`, but if we've reach the maximum degree
      # then we need to move to the next index down

      # if the current tuple ν has admissible degree ...
      if isadmissible
         # ... then we add it to the stack  ...
         #     (at least if it is an admissible basis function respecting
         #      all the symmetries - this is checked by filter_tuple)
         if !filter || filter_tuple(ll, cg)
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
   return nothing
end
