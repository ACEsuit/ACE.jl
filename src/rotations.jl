
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module Rotations

using StaticArrays
using LinearAlgebra: norm, rank, svd, Diagonal
using SHIPs: _mrange, IntS
using SHIPs.SphericalHarmonics: index_y

export ClebschGordan, CoeffArray, single_B


# TODO: reduce dimensionality of the storage tensor
#       to numY x numY x maxL (or possibly less?)
"""
`ClebschGordan: ` storing precomputed Clebsch-Gordan coefficients; see
`?clebschgordan` for the convention that is use.
"""
struct ClebschGordan{T}
	# maxL::Int
	# cg::Array{T, 3}   # rewrite as Dict!
	vals::Dict{Tuple{IntS, IntS, IntS}, T}
end

"""
`CoeffArray: ` storing recursively precomputed coefficients for a
rotation-invariant basis.
"""
struct CoeffArray{T}
   vals::Vector{Dict}
   cg::ClebschGordan{T}
end


# ----------------------------------------------------------------------
#     ClebschGordan code
# ----------------------------------------------------------------------


cg_conditions(j1,m1,j2,m2,j3,m3) =
	cg_l_condition(j1, j2, j3) &&
	cg_m_condition(m1, m2, m3)   &&
	(abs(m1) <= j1) && (abs(m2) <= j2) && (abs(m3) <= j3)

cg_l_condition(j1, j2, j3) = (abs(j1-j2) <= j3 <= j1 + j2)
cg_m_condition(m1, m2, m3) = (m3 == m1 + m2)


"""
`clebschgordan(j1, m1, j2, m2, j3, m3, T=Float64)` :

A reference implementation of Clebsch-Gordon coefficients based on

https://hal.inria.fr/hal-01851097/document
Equation (4-6)

This heavily uses BigInt and BigFloat and should therefore not be employed
for performance critical tasks.

The ordering of parameters corresponds to the following convention:
```
clebschgordan(j1, m1, j2, m2, j3, m3) = C_{j1m1j2m2}^{j3m3}
```
where
```
   D_{m1k1}^{l1} D_{m2k2}^{l2}}
	=
	∑_j  C_{l1m1l2m2}^{j(m1+m2)} C_{l1k1l2k2}^{j2(k1+k2)} D_{(m1+m2)(k1+k2)}^{j}
```
"""
function clebschgordan(j1, m1, j2, m2, j3, m3, T=Float64)
	if !cg_conditions(j1, m1, j2, m2, j3, m3)
		return zero(T)
	end

   N = (2*j3+1) *
       factorial(big(j1+m1)) * factorial(big(j1-m1)) *
       factorial(big(j2+m2)) * factorial(big(j2-m2)) *
       factorial(big(j3+m3)) * factorial(big(j3-m3)) /
       factorial(big( j1+j2-j3)) /
       factorial(big( j1-j2+j3)) /
       factorial(big(-j1+j2+j3)) /
       factorial(big(j1+j2+j3+1))

   G = big(0)
   # 0 ≦ k ≦ j1+j2-j3
   # 0 ≤ j1-m1-k ≤ j1-j2+j3   <=>   j2-j3-m1 ≤ k ≤ j1-m1
   # 0 ≤ j2+m2-k ≤ -j1+j2+j3  <=>   j1-j3+m2 ≤ k ≤ j2+m2
   lb = (0, j2-j3-m1, j1-j3+m2)
   ub = (j1+j2-j3, j1-m1, j2+m2)
   for k in maximum(lb):minimum(ub)
      bk = big(k)
      G += (-1)^k *
           binomial(big( j1+j2-j3), big(k)) *
           binomial(big( j1-j2+j3), big(j1-m1-k)) *
           binomial(big(-j1+j2+j3), big(j2+m2-k))
   end

   return T(sqrt(N) * G)
end


ClebschGordan(T=Float64) =
	ClebschGordan(Dict{Tuple{IntS,IntS,IntS}, T}())

_cg_key(j1, m1, j2, m2, j3, m3) =
	IntS.((index_y(j1,m1), index_y(j2,m2), index_y(j3,m3)))

function (cg::ClebschGordan{T})(j1, m1, j2, m2, j3, m3) where {T}
	if !cg_conditions(j1,m1,j2,m2,j3,m3)
		return zero(T)
	end
	key = _cg_key(j1, m1, j2, m2, j3, m3)
	if haskey(cg.vals, key)
		return cg.vals[key]
	end
	val = clebschgordan(j1, m1, j2, m2, j3, m3, T)
	cg.vals[key] = val
	return val
end


# ----------------------------------------------------------------------
#     CoeffArray code
# ----------------------------------------------------------------------

dicttype(N::Integer) = dicttype(Val(N))

dicttype(::Val{N}) where {N} =
   Dict{Tuple{SVector{N,Int8}, SVector{N,Int8}, SVector{N,Int8}}, Float64}

CoeffArray(T=Float64) = CoeffArray(Dict[], ClebschGordan(T))


function get_vals(A::CoeffArray, valN::Val{N}) where {N}
	if length(A.vals) < N
		for n = length(A.vals)+1:N
			push!(A.vals, dicttype(n)())
		end
	end
   return A.vals[N]::dicttype(valN)
end

_key(ll::StaticVector{N}, mm::StaticVector{N}, kk::StaticVector{N}) where {N} =
      (SVector{N, Int8}(ll), SVector{N, Int8}(mm), SVector{N, Int8}(kk))

function (A::CoeffArray)(ll::StaticVector{N},
                         mm::StaticVector{N},
                         kk::StaticVector{N}) where {N}
   if       sum(mm) != 0 ||
            sum(kk) != 0 ||
            !all(abs.(mm) .<=  ll) ||
            !all(abs.(kk) .<= ll)
      return 0.0
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

function (A::CoeffArray)(ll::StaticVector{1},
                         mm::StaticVector{1},
                         kk::StaticVector{1})
   if ll[1] == mm[1] == kk[1] == 0
      return 1.0
   else
      return 0.0
   end
end

function (A::CoeffArray)(ll::StaticVector{2},
                         mm::StaticVector{2},
                         kk::StaticVector{2})
   if ll[1] != ll[2] || sum(mm) != 0 || sum(kk) != 0
      return 0.0
   else
      return 8 * pi^2 / (2*ll[1]+1) * (-1)^(mm[1]-kk[1])
   end
end


function _compute_val(A::CoeffArray, ll::StaticVector{N},
                                     mm::StaticVector{N},
                                     kk::StaticVector{N}) where {N}
	val = 0.0
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
			0.0
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


_len_mrange(ll) = sum(_ -> 1, _mrange(ll))


function basis(A::CoeffArray{T}, ll) where {T}
	len = _len_mrange(ll)
	CC = compute_Al(A, ll)
	svdC = svd(CC)
	rk = rank(Diagonal(svdC.S))
	return svdC.U[:, 1:rk]
end


compute_Al(ll::SVector{N}) where {N} = compute_Al(CoeffArray(N, sum(ll)), ll)


function compute_Al(A::CoeffArray{T}, ll::SVector) where {T}
	len = _len_mrange(ll)
   CC = zeros(T, len, len)
   for (im, mm) in enumerate(_mrange(ll)), (ik, kk) in enumerate(_mrange(ll))
      CC[ik, im] = A(ll, mm, kk)
   end
   return CC
end


function single_B(A::CoeffArray{T}, ll::SVector) where {T}
	MM = collect(_mrange(ll))
	Is = sortperm([ (sum(mm .!= 0), norm(mm)) for mm in MM ])
	MM = MM[Is]
	CC = zeros(T, length(MM))
	for mm in MM
		for (ik, kk) in enumerate(_mrange(ll))
			CC[ik] = A(ll, mm, kk)
		end
		if norm(CC) > 0
			CC ./= norm(CC)
			break
		end
	end
	return CC
end


end
