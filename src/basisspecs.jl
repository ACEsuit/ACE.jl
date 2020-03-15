
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# This file implements different ways to specify a degree

using Combinatorics
using JuLIP.Chemistry: atomic_number
import JuLIP.Potentials: z2i, i2z

import Base: ==

export SparseSHIP



abstract type BasisSpec{BO, NZ} end
abstract type AnalyticBasisSpec{BO, NZ} <: BasisSpec{BO, NZ} end

get_filter(::AnalyticBasisSpec) = (_ -> true)

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


z2i(spec::AnalyticBasisSpec, z::Integer) = spec.z2i[z]

i2z(spec::AnalyticBasisSpec, z::Integer) = spec.Zs[i]

nspecies(spec::AnalyticBasisSpec) = length(spec.Zs)

admissible(D::AnalyticBasisSpec, k, l) = deg(D, k, l) <= D.deg



"""
`SparseSHIP` : a general sparse-grid type degree definition,
```
deg({k}, {l}) = csp ∑ᵢ (kᵢ + wL * lᵢ)
              + chc ∏ᵢ max(ahc, bhc + kᵢ + wL * lᵢ)
```

Constructors:
```
SparseSHIP(bo, deg; kwargs...)
SparseSHIP(Zs, bo, deg; kwargs...)
```
where the keyword arguments are (with defaults)
* `wL = 1.5`
* `csp = 1.0`
* `chc = 0.0`
* `ahc = 0.0`
* `bhc = 0.0`
"""
struct SparseSHIP{BO, NZ} <: AnalyticBasisSpec{BO, NZ}
   deg::IntS
   wL::Float64
   csp::Float64
   chc::Float64
   ahc::Float64
   bhc::Float64
   # --------------------
   Zs::NTuple{NZ, Int16}
   valbo::Val{BO}
   z2i::Dict{Int16, Int16}
   # --------------------
   filterfcn
end

get_filter(spec::SparseSHIP) = spec.filterfcn

==(s1::SparseSHIP, s2::SparseSHIP) =
      all( getfield(s1, i) == getfield(s2, i)
           for i = 1:fieldcount(SparseSHIP)-1 )

SparseSHIP(bo::Integer, deg::Integer; kwargs...) =
      SparseSHIP(:X, bo, deg; kwargs...)

function SparseSHIP(Zs, bo::Integer, deg::Integer;
                    wL = 1.5,
                    csp = 1.0,
                    chc = 0.0,
                    ahc = 0.0,
                    bhc = 0.0,
                    filterfcn = _ -> true)
   @assert wL > 0
   @assert deg > 0
   @assert bo >= 0
   Zs = _convert_Zs(Zs)
   z2i = Dict([ Int16(z) => Int16(i) for (i, z) in enumerate(Zs) ]...)
   return SparseSHIP(IntS(deg), Float64(wL), Float64(csp),
                     Float64(chc), Float64(ahc), Float64(bhc),
                     Zs, Val(bo), z2i,
                     filterfcn )
end


deg(D::SparseSHIP, k::Integer, l::Integer) =
      k + D.wL * l

deg(D::SparseSHIP, kk::VecOrTup, ll::VecOrTup) =
      D.csp * sum( deg(D, k, l) for (k, l) in zip(kk, ll) ) +
      D.chc * prod( max(D.ahc, D.bhc + deg(D, k, l)) for (k, l) in zip(kk, ll) )

# if one k is non-zero all other ks and ls are zero then we get
#    csp * k + chc * max(ahc, bhc)^{N-1} * max(ahc, bhc + k)
# take body-order = 1 (minimal) then we get just
#    csp * k + chc * max(ahc, bhc + k)
function maxK(D::SparseSHIP)
   allk = 0:ceil(Int, D.deg / D.csp)
   # degs = D.csp * allk + D.chc * max.(ahc, bhc .+ allk)
   degs = [ deg(D, (k,), (0,)) for k in allk ]
   admissible = findall(degs .<= D.deg)
   return maximum(allk[admissible])
end

# For a pure 2-body potential we don't need an angular component
maxL(D::SparseSHIP{1}, k::Integer =  0) = 0

# if just one k is non-zero (or 0), just one l non-zero and all other
# ks and ls are zero then we get
#    csp * (k+wL*l) + chc * max(ahc, bhc + k + wL*l)
function maxL(D::SparseSHIP, k::Integer = 0)
   alll = 0:ceil(Int, (D.deg / D.csp - k) / D.wL)
   degs = [ deg(D, (k,), (l,)) for l in alll ]
   admissible = findall(degs .<= D.deg)
   if isempty(admissible)
      return 0
   else
      return maximum(alll[admissible])
   end
end

Dict(D::SparseSHIP{BO}) where {BO} = Dict("__id__" => "SHIPs_SparseSHIP",
                                "deg" => D.deg,
                                "wL" => D.wL,
                                "csp" => D.csp,
                                "chc" => D.chc,
                                "ahc" => D.ahc,
                                "bhc" => D.bhc,
                                "Zs" => D.Zs,
                                "bo" => BO)

convert(::Val{:SHIPs_SparseSHIP}, D::Dict) =
      SparseSHIP(D["Zs"], D["bo"], D["deg"],
                 wL = D["wL"], csp = D["csp"],
                 chc = D["chc"], ahc = D["ahc"], bhc = D["bhc"] )



# ---------------------------------------------------------------
#    generating the basis specification
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
   return allZKL
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

function _kl(ν::StaticVector{N}, izz::StaticVector{N}, KLZ) where {N}
   kk = SVector( ntuple(α -> KLZ[izz[α]][ν[α]].k, N) )
   ll = SVector( ntuple(α -> KLZ[izz[α]][ν[α]].l, N) )
   return kk, ll
end

function _kl2tup(args...)
   kk, ll = _kl(args...)
   return (kk = kk, ll = ll)
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
   return SMatrix{bo,nz,Vector}(Nu)
end


function generate_ZKL_tuples(spec::AnalyticBasisSpec{BO}, rotcoefs;
                            filter=true) where {BO}
   # all possible (k, l) pairs
   allZKL = generate_ZKL(spec)
   # separate arrays for all body-orders and species
   NuZ = _init_NuZ(BO, nspecies(spec))
   for N = 1:BO
      _generate_ZKL_tuples!(NuZ[N,1], spec, rotcoefs, allZKL, Val(N);
                            filter=filter, filterfcn=get_filter(spec))
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


function _generate_ZKL_tuples!(NuZ, spec::AnalyticBasisSpec, rotcoefs, ZKL, ::Val{BO};
                               filter=true, filterfcn = _->true) where {BO}
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
      _generate_KL_tuples!(Nu, spec, rotcoefs, ZKL[izz]; filter=filter)

      # now loop through all the ν tuples we found to push them into NuZ
      for ν in Nu
         # for each such ν we also need to check whether any permutations
         # of `izz` give a new basis function as well. E.g.,
         #   A[1,n] A[1,m] == A[1,m] A[1,n]
         #   A[1,n] A[2,n] == A[2,n] A[1,n]
         # but
         #   A[1,n] A[2,m] != A[2,n] A[1,m]
         empty!(izz_tmp)
         if filterfcn(_kl2tup(ν, izz, ZKL))
            push!(izz_tmp, izz)
            push!(NuZ, (izz = izz, ν = ν))
         end
         for izzp in unique(permutations(izz))
            if !any( _iseqB(SVector(izzp...), ν, izz1, ν)  for izz1 in izz_tmp )
               # and finally a user-defined filter, this is currently used
               # for the orth-to-zero basis, but could be used in many ways...
               if filterfcn(_kl2tup(ν, izz, ZKL))
                  push!(izz_tmp, izzp)
                  push!(NuZ, (izz = izzp, ν = ν))
               end
            end
         end
      end
   end
end


function _generate_KL_tuples!(Nu::Vector{<: SVector{BO}},
                              spec::AnalyticBasisSpec,
                              rotcoefs, ZKLs;
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
         if !filter || filter_tuple(ll, rotcoefs)
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


function filter_tuple(ll, rotcoefs)
   if isodd(sum(ll))
      return false
   end
   Bcoefs = SHIPs.Rotations.single_B(rotcoefs, ll)
   return norm(Bcoefs) > 1e-12
end
