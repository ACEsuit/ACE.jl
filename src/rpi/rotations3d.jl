
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



module Rotations3D

using StaticArrays
using LinearAlgebra: norm, rank, svd, Diagonal
using ACE.SphericalHarmonics: index_y
using Combinatorics: permutations

export ClebschGordan, Rot3DCoeffs, ri_basis, rpi_basis


"""
`ClebschGordan: ` storing precomputed Clebsch-Gordan coefficients; see
`?clebschgordan` for the convention that is use.
"""
struct ClebschGordan{T}
	vals::Dict{Tuple{Int, Int, Int, Int, Int, Int}, T}
end

"""
`Rot3DCoeffs: ` storing recursively precomputed coefficients for a
rotation-invariant basis.
"""
struct Rot3DCoeffs{T}
   vals::Vector{Dict}
   cg::ClebschGordan{T}
end



# -----------------------------------
# iterating over an m collection
# -----------------------------------

_mvec(::CartesianIndex{0}) = SVector(Int(0))

_mvec(mpre::CartesianIndex) = SVector(Tuple(mpre)..., - sum(Tuple(mpre)))

struct MRange{N, T2}
   ll::SVector{N, Int}
   cartrg::T2
end

Base.length(mr::MRange) = sum(_->1, _mrange(mr.ll))

"""
Given an l-vector `ll` iterate over all combinations of `mm` vectors  of
the same length such that `sum(mm) == 0`
"""
_mrange(ll) = MRange(ll, Iterators.Stateful(
                     CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll)-1))))

function Base.iterate(mr::MRange{1}, args...)
   if isempty(mr.cartrg)
      return nothing
   end
   while !isempty(mr.cartrg)
      popfirst!(mr.cartrg)
   end
   return SVector{1, Int}(0), nothing
end

function Base.iterate(mr::MRange, args...)
   while true
      if isempty(mr.cartrg)
         return nothing
      end
      mpre = popfirst!(mr.cartrg)
      if abs(sum(mpre.I)) <= mr.ll[end]
         return _mvec(mpre), nothing
      end
   end
   error("we should never be here")
end



# ----------------------------------------------------------------------
#     ClebschGordan code
# ----------------------------------------------------------------------


cg_conditions(j1,m1, j2,m2, J,M) =
	cg_l_condition(j1, j2, J)   &&
	cg_m_condition(m1, m2, M)   &&
	(abs(m1) <= j1) && (abs(m2) <= j2) && (abs(M) <= J)

cg_l_condition(j1, j2, J) = (abs(j1-j2) <= J <= j1 + j2)

cg_m_condition(m1, m2, M) = (M == m1 + m2)


"""
`clebschgordan(j1, m1, j2, m2, J, M, T=Float64)` :

A reference implementation of Clebsch-Gordon coefficients based on

https://hal.inria.fr/hal-01851097/document
Equation (4-6)

This heavily uses BigInt and BigFloat and should therefore not be employed
for performance critical tasks, but only precomputation.

The ordering of parameters corresponds to the following convention:
```
clebschgordan(j1, m1, j2, m2, J, M) = C_{j1m1j2m2}^{JM}
```
where
```
   D_{m1k1}^{l1} D_{m2k2}^{l2}}
	=
	∑_j  C_{l1m1l2m2}^{j(m1+m2)} C_{l1k1l2k2}^{j2(k1+k2)} D_{(m1+m2)(k1+k2)}^{j}
```
"""
function clebschgordan(j1, m1, j2, m2, J, M, T=Float64)
	if !cg_conditions(j1, m1, j2, m2, J, M)
		return zero(T)
	end

   N = (2*J+1) *
       factorial(big(j1+m1)) * factorial(big(j1-m1)) *
       factorial(big(j2+m2)) * factorial(big(j2-m2)) *
       factorial(big(J+M)) * factorial(big(J-M)) /
       factorial(big( j1+j2-J)) /
       factorial(big( j1-j2+J)) /
       factorial(big(-j1+j2+J)) /
       factorial(big(j1+j2+J+1))

   G = big(0)
   # 0 ≦ k ≦ j1+j2-J
   # 0 ≤ j1-m1-k ≤ j1-j2+J   <=>   j2-J-m1 ≤ k ≤ j1-m1
   # 0 ≤ j2+m2-k ≤ -j1+j2+J  <=>   j1-J+m2 ≤ k ≤ j2+m2
   lb = (0, j2-J-m1, j1-J+m2)
   ub = (j1+j2-J, j1-m1, j2+m2)
   for k in maximum(lb):minimum(ub)
      bk = big(k)
      G += (-1)^k *
           binomial(big( j1+j2-J), big(k)) *
           binomial(big( j1-j2+J), big(j1-m1-k)) *
           binomial(big(-j1+j2+J), big(j2+m2-k))
   end

   return T(sqrt(N) * G)
end


ClebschGordan(T=Float64) =
	ClebschGordan{T}(Dict{Tuple{Int,Int,Int,Int,Int,Int}, T}())

_cg_key(j1, m1, j2, m2, J, M) = (j1, m1, j2, m2, J, M)
	# Int.((index_y(j1,m1), index_y(j2,m2), index_y(J,M)))

function (cg::ClebschGordan{T})(j1, m1, j2, m2, J, M) where {T}
	if !cg_conditions(j1,m1, j2,m2, J,M)
		return zero(T)
	end
	key = _cg_key(j1, m1, j2, m2, J, M)
	if haskey(cg.vals, key)
		return cg.vals[key]
	end
	val = clebschgordan(j1, m1, j2, m2, J, M, T)
	cg.vals[key] = val
	return val
end


# ----------------------------------------------------------------------
#     Rot3DCoeffs code: generalized cg coefficients
#
#  Note: in this section kk is a tuple of m-values, it is not
#        related to the k index in the 1-p basis (or radial basis)
# ----------------------------------------------------------------------

dicttype(N::Integer) = dicttype(Val(N))

dicttype(::Val{N}) where {N} =
   Dict{Tuple{SVector{N,Int}, SVector{N,Int}, SVector{N,Int}}, Float64}

Rot3DCoeffs(T=Float64) = Rot3DCoeffs(Dict[], ClebschGordan(T))


function get_vals(A::Rot3DCoeffs, valN::Val{N}) where {N}
	if length(A.vals) < N
		for n = length(A.vals)+1:N
			push!(A.vals, dicttype(n)())
		end
	end
   return A.vals[N]::dicttype(valN)
end

_key(ll::StaticVector{N}, mm::StaticVector{N}, kk::StaticVector{N}) where {N} =
      (SVector{N, Int}(ll), SVector{N, Int}(mm), SVector{N, Int}(kk))

function (A::Rot3DCoeffs{T})(ll::StaticVector{N},
                            mm::StaticVector{N},
                            kk::StaticVector{N}) where {T, N}
   if       sum(mm) != 0 ||
            sum(kk) != 0 ||
            !all(abs.(mm) .<= ll) ||
            !all(abs.(kk) .<= ll)
      return T(0)
   end
   vals = get_vals(A, Val(N))  # this should infer the type!
   key = _key(ll, mm, kk)
   if haskey(vals, key)
      val  = vals[key]
   else
      val = _compute_val(A, key...)
      vals[key] = val
   end
   return val
end

# the recursion has two steps so we need to define the
# coupling coefficients for N = 1, 2
# TODO: actually this seems false; it is only one recursion step, and a bit
#       or reshuffling should allow us to get rid of the {N = 2} case.

function (A::Rot3DCoeffs{T})(ll::StaticVector{1},
                            mm::StaticVector{1},
                            kk::StaticVector{1}) where {T}
   if ll[1] == mm[1] == kk[1] == 0
      return T(1)
   else
      return T(0)
   end
end

function (A::Rot3DCoeffs{T})(ll::StaticVector{2},
                            mm::StaticVector{2},
                            kk::StaticVector{2}) where {T}
   if ll[1] != ll[2] || sum(mm) != 0 || sum(kk) != 0
      return T(0)
   else
      return T( 8 * pi^2 / (2*ll[1]+1) * (-1)^(mm[1]-kk[1]) )
   end
end

# next comes the recursion step for N ≧ 3

function _compute_val(A::Rot3DCoeffs{T}, ll::StaticVector{N},
                                        mm::StaticVector{N},
                                        kk::StaticVector{N}) where {T, N}
	val = T(0)
   llp = ll[1:N-2]
   mmp = mm[1:N-2]
   kkp = kk[1:N-2]
   for j = abs(ll[N-1]-ll[N]):(ll[N-1]+ll[N])
      if abs(kk[N-1]+kk[N]) > j || abs(mm[N-1]+mm[N]) > j
         continue
      end
		cgk = try
			A.cg(ll[N-1], kk[N-1], ll[N], kk[N], j, kk[N-1]+kk[N])
		catch
			@show (ll[N-1], kk[N-1], ll[N], kk[N], j, kk[N-1]+kk[N])
			T(0)
		end
		cgm = A.cg(ll[N-1], mm[N-1], ll[N], mm[N], j, mm[N-1]+mm[N])
		if cgk * cgm  != 0
			val += cgk * cgm * A( SVector(llp..., j),
								       SVector(mmp..., mm[N-1]+mm[N]),
								       SVector(kkp..., kk[N-1]+kk[N]) )
		end
   end
   return val
end


# ----------------------------------------------------------------------
#   construction of a possible set of generalised CG coefficient;
#   numerically via SVD
# ----------------------------------------------------------------------


function ri_basis(A::Rot3DCoeffs{T}, ll::SVector; ordered=false) where {T}
	CC = compute_Al(A, ll, Val(ordered))
	svdC = svd(CC)
	rk = rank(Diagonal(svdC.S))
	return svdC.U[:, 1:rk]'
end


# unordered
function compute_Al(A::Rot3DCoeffs{T}, ll::SVector, ::Val{false}) where {T}
	len = length(_mrange(ll))
   CC = zeros(T, len, len)
   for (im, mm) in enumerate(_mrange(ll)), (ik, kk) in enumerate(_mrange(ll))
      CC[ik, im] = A(ll, mm, kk)
   end
   return CC
end

# # ordered; TODO: check this out, clean it up and test it!!!
# function compute_Al(A::Rot3DCoeffs{T}, ll::SVector, ::Val{true}) where {T}
# 	num_mm_sorted = sum(mm -> issorted(mm), _mrange(ll))
# 	# @show num_mm_sorted
# 	num_mm = length(_mrange(ll))
#    CC = zeros(T, num_mm, num_mm_sorted)
# 	im = 0
#    for mm in _mrange(ll)
# 		if issorted(mm) # -> make this sorted relative to ll!!!
# 			im += 1
# 			for (ik, kk) in enumerate(_mrange(ll))
# 		      CC[ik, im] = A(ll, mm, kk)
# 			end
# 		end
# 	end
#    return CC
# end
#
#
# # two utility functions which are probably never used!
#
# compute_Al(ll::SVector{N}; ordered = false) where {N} =
# 		compute_Al(Rot3DCoeffs(N, sum(ll)), ll; ordered=ordered)
#
# compute_Al(A::Rot3DCoeffs, ll::SVector{N}; ordered = false) where {N} =
# 		compute_Al(A, ll, Val(ordered))


# TODO: this could use some documentation

rpi_basis(A::Rot3DCoeffs, zz, nn, ll) =
			rpi_basis(A, SVector(zz...), SVector(nn...), SVector(ll...))

function rpi_basis(A::Rot3DCoeffs,
						 zz::SVector{N},
						 nn::SVector{N, Int},
						 ll::SVector{N, Int}) where {N}
	Uri = ri_basis(A, ll)
	Mri = collect( _mrange(ll) )   # rows...
	G = _gramian(zz, nn, ll, Uri, Mri)
   S = svd(G)
   rk = rank(G; rtol =  1e-7)
	Urpi = S.U[:, 1:rk]'
	return Diagonal(sqrt.(S.S[1:rk])) * Urpi * Uri, Mri
end


function _gramian(zz, nn, ll, Uri, Mri)
   N = length(nn)
   nri = size(Uri, 1)
   @assert size(Uri, 1) == nri
   G = zeros(nri, nri)
   for σ in permutations(1:N)
      if (zz[σ] != zz) || (nn[σ] != nn) || (ll[σ] != ll); continue; end
      for (iU1, mm1) in enumerate(Mri), (iU2, mm2) in enumerate(Mri)
         if mm1[σ] == mm2
            for i1 = 1:nri, i2 = 1:nri
               G[i1, i2] += conj(Uri[i1, iU1]) * Uri[i2, iU2]
            end
         end
      end
   end
   return G
end



end
