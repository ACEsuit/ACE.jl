
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module SphericalHarmonics

using StaticArrays, LinearAlgebra

import SHIPs

import SHIPs: alloc_B, alloc_dB, alloc_temp, alloc_temp_d,
              eval_basis!, eval_basis_d!

const JVec = SVector{3}

export SHBasis, ClebschGordan


struct PseudoSpherical{T}
	r::T
	cosφ::T
	sinφ::T
	cosθ::T
	sinθ::T
end

spher2cart(S::PseudoSpherical) = S.r * JVec(S.cosφ*S.sinθ, S.sinφ*S.sinθ, S.cosθ)

function cart2spher(R::AbstractVector)
	r = norm(R)
	φ = atan(R[2], R[1])
	sinφ, cosφ = sincos(φ)
	cosθ = R[3] / r
	sinθ = sqrt(R[1]^2+R[2]^2) / r
	return PseudoSpherical(r, cosφ, sinφ, cosθ, sinθ)
end

PseudoSpherical(φ, θ) = PseudoSpherical(1.0, cos(φ), sin(φ), cos(θ), sin(θ))

"""
convert a gradient with respect to spherical coordinates to a gradient
with respect to cartesian coordinates
"""
function dspher_to_dcart(S, f_φ_div_sinθ, f_θ)
	r = S.r + eps()
   return SVector( - (S.sinφ * f_φ_div_sinθ) + (S.cosφ * S.cosθ * f_θ),
			            (S.cosφ * f_φ_div_sinθ) + (S.sinφ * S.cosθ * f_θ),
			 			                                 - (   S.sinθ * f_θ) ) / r
end

# function dspher_to_dcart_m0(S, f_θ)
# 	r = S.r + eps()
#    return SVector(S.cosφ * S.cosθ, S.sinφ * S.cosθ, S.sinθ) * (f_θ / r)
# end



# --------------------------------------------------------
#     Indexing
# --------------------------------------------------------

"""
	sizeP(maxDegree)

Return the size of the set of Associated Legendre Polynomials ``P_l^m(x)`` of
degree less than or equal to the given maximum degree
"""
sizeP(maxDegree) = div((maxDegree + 1) * (maxDegree + 2), 2)

"""
	sizeY(maxDegree)

Return the size of the set of real spherical harmonics ``Y_{l,m}(θ,φ)`` of
degree less than or equal to the given maximum degree
"""
sizeY(maxDegree) = (maxDegree + 1) * (maxDegree + 1)

"""
	index_p(l,m)

Return the index into a flat array of Associated Legendre Polynomials ``P_l^m``
for the given indices ``(l,m)``.
``P_l^m`` are stored in l-major order i.e. [P(0,0), [P(1,0), P(1,1), P(2,0), ...]
"""
# Base.@pure index_p(l,m) = m + div(l*(l+1), 2) + 1
@inline index_p(l,m) = m + div(l*(l+1), 2) + 1

"""
	index_y(l,m)

Return the index into a flat array of real spherical harmonics ``Y_{l,m}``
for the given indices ``(l,m)``.
``Y_{l,m}`` are stored in l-major order i.e.
[Y(0,0), [Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), ...]
"""
Base.@pure index_y(l,m) = m + l + (l*l) + 1


# --------------------------------------------------------
#     Associated Legendre Polynomials
#     TODO: rewrite within general interface?
#           - alloc_B, alloc_dB, alloc_temp, ...
# --------------------------------------------------------

"""
TODO: documentation
"""
struct ALPCoefficients{T}
	A::Vector{T}
	B::Vector{T}
end

ALPCoefficients(maxDegree::Integer, T=Float64) =
	ALPCoefficients( Vector{T}(undef, sizeP(maxDegree)),
						  Vector{T}(undef, sizeP(maxDegree)) )

"""
	compute_coefficients(L)

Precompute coefficients ``a_l^m`` and ``b_l^m`` for all l <= L, m <= l
"""
function compute_coefficients(L::Integer)
	coeff = ALPCoefficients(L)
	for l in 2:L
		ls = l*l
		lm1s = (l-1) * (l-1)
		for m in 0:(l-2)
			ms = m * m
			coeff.A[index_p(l, m)] = sqrt((4 * ls - 1.0) / (ls - ms))
			coeff.B[index_p(l, m)] = -sqrt((lm1s - ms) / (4 * lm1s - 1.0))
		end
	end
	return coeff
end

"""
	compute_coefficients(L)

Create an array large enough to store an entire set of Associated Legendre
Polynomials ``P_l^m(x)`` of maximum degree L.
"""
allocate_p(L::Int) = Array{Float64}(undef, sizeP(L))

"""
	compute_p(L, x, coeff, P)

Compute an entire set of Associated Legendre Polynomials ``P_l^m(x)``
using the given coefficients, and store in the array P.
"""
function compute_p!(L::Integer, S::PseudoSpherical{T}, coeff::ALPCoefficients{T},
					     P::Array{T,1}) where {T}
   @assert L > 0
	@assert length(coeff.A) >= sizeP(L)
	@assert length(coeff.B) >= sizeP(L)
	@assert length(P) >= sizeP(L)

	temp = sqrt(0.5/π)
	P[index_p(0, 0)] = temp

	if (L > 0)
		P[index_p(1, 0)] = S.cosθ * sqrt(3) * temp
		temp = - sqrt(1.5) * S.sinθ * temp
		P[index_p(1, 1)] = temp

		for l in 2:L
			il = ((l*(l+1)) ÷ 2) + 1
			ilm1 = il - l
			ilm2 = ilm1 - l + 1
			for m in 0:(l-2)
				# FOR DEBUGGING TURN ON THESE ASSERTS!
				# @assert il+m == index_p(l, m)
				# @assert ilm1+m == index_p(l-1,m)
				# @assert ilm2+m == index_p(l-2,m)
				@inbounds P[il+m] = coeff.A[il+m] * (     S.cosθ * P[ilm1+m]
						                           + coeff.B[il+m] * P[ilm2+m] )
			end
			@inbounds P[il+l-1] = S.cosθ * sqrt(2 * (l - 1) + 3) * temp
			temp = -sqrt(1.0 + 0.5 / l) * S.sinθ * temp
			@inbounds P[il+l] = temp
		end
	end
	return P
end


function compute_dp!(L::Integer, S::PseudoSpherical{T}, coeff::ALPCoefficients{T},
					     P::Array{T,1}, dP::Array{T,1}) where T
   @assert L > 0
	@assert length(coeff.A) >= sizeP(L)
	@assert length(coeff.B) >= sizeP(L)
	@assert length(P) >= sizeP(L)

	temp = sqrt(0.5/π)
	P[index_p(0, 0)] = temp
	temp_d = 0.0
	dP[index_p(0, 0)] = temp_d

	if (L > 0)
		P[index_p(1, 0)] = S.cosθ * sqrt(3) * temp
		dP[index_p(1, 0)] = -S.sinθ * sqrt(3) * temp + S.cosθ * sqrt(3) * temp_d

		temp1, temp_d = ( - sqrt(1.5) * temp,
						      - sqrt(1.5) * (S.cosθ * temp + S.sinθ * temp_d) )
		P[index_p(1, 1)] = temp1
		dP[index_p(1, 1)] = temp_d

		for l in 2:L
			m = 0
			@inbounds P[index_p(l, m)] =
					coeff.A[index_p(l, m)] * (     S.cosθ * P[index_p(l - 1, m)]
					             + coeff.B[index_p(l, m)] * P[index_p(l - 2, m)] )
			@inbounds dP[index_p(l, m)] =
				coeff.A[index_p(l, m)] * (
								- S.sinθ * P[index_p(l - 1, m)]
								+ S.cosθ * dP[index_p(l - 1, m)]
				             + coeff.B[index_p(l, m)] * dP[index_p(l - 2, m)] )

			for m in 1:(l-2)
				@inbounds P[index_p(l, m)] =
						coeff.A[index_p(l, m)] * (     S.cosθ * P[index_p(l - 1, m)]
						             + coeff.B[index_p(l, m)] * P[index_p(l - 2, m)] )
				@inbounds dP[index_p(l, m)] =
					coeff.A[index_p(l, m)] * (
									- S.sinθ^2 * P[index_p(l - 1, m)]
									+ S.cosθ * dP[index_p(l - 1, m)]
					             + coeff.B[index_p(l, m)] * dP[index_p(l - 2, m)] )
			end
			@inbounds P[index_p(l, l - 1)] = sqrt(2 * (l - 1) + 3) * S.cosθ * temp1
			@inbounds dP[index_p(l, l - 1)] = sqrt(2 * (l - 1) + 3) * (
										        -S.sinθ^2 * temp1 + S.cosθ * temp_d )

         (temp1, temp_d) = (
						-sqrt(1.0+0.5/l) * S.sinθ * temp1,
			         -sqrt(1.0+0.5/l) * (S.cosθ * temp1 * S.sinθ + S.sinθ * temp_d) )
			@inbounds P[index_p(l, l)] = temp1
			@inbounds dP[index_p(l, l)] = temp_d
		end
	end
	# return P, dP
end





"""
	compute_p(L, x)

Compute an entire set of Associated Legendre Polynomials ``P_l^m(x)`` where
``0 ≤ l ≤ L`` and ``0 ≤ m ≤ l``. Assumes ``|x| ≤ 1``.
"""
function compute_p(L::Integer, S::PseudoSpherical{T}) where {T}
	P = Array{T}(undef, sizeP(L))
	coeff = compute_coefficients(L)
	compute_p!(L, S, coeff, P)
	return P
end

function compute_dp(L::Integer, S::PseudoSpherical{T}) where {T}
	P = Array{T}(undef, sizeP(L))
	dP = Array{T}(undef, sizeP(L))
	coeff = compute_coefficients(L)
	compute_dp!(L, S, coeff, P, dP)
	return P, dP
end

compute_p(L::Integer, θ::Real) =
	compute_p(L, PseudoSpherical(0.0, 0.0, 0.0, cos(θ), sin(θ)))

compute_dp(L::Integer, θ::Real) =
	compute_dp(L, PseudoSpherical(0.0, 0.0, 0.0, cos(θ), sin(θ)))



# ------------------------------------------------------------------------
#                  Spherical Harmonics
# ------------------------------------------------------------------------


function cYlm!(Y, L, S::PseudoSpherical, P)
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
   @assert abs(S.cosθ) <= 1.0

	ep = 1 / sqrt(2)
	for l = 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * ep
	end

   sig = 1
   ep_fact = S.cosφ + im * S.sinφ
	for m in 1:L
		sig *= -1
		ep *= ep_fact            # ep =   exp(i *   m  * φ)
		em = sig * conj(ep)      # ep = ± exp(i * (-m) * φ)
		for l in m:L
			p = P[index_p(l,m)]
			@inbounds Y[index_y(l, -m)] = em * p   # (-1)^m * p * exp(-im*m*phi) / sqrt(2)
			@inbounds Y[index_y(l,  m)] = ep * p   #          p * exp( im*m*phi) / sqrt(2)
		end
	end

	return Y
end


function cYlm_d!(Y, dY, L, S::PseudoSpherical, P, dP)
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
	@assert length(dY) >= sizeY(L)
   # @assert abs(S.cosθ) < 1.0

	# m = 0 case
	ep = 1 / sqrt(2)
	for l = 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * ep
		# dY[index_y(l, 0)] = dspher_to_dcart_m0(S, dP[index_p(l, 0)] * ep)
		dY[index_y(l, 0)] = dspher_to_dcart(S, 0.0, dP[index_p(l, 0)] * ep)
	end

   sig = 1
   ep_fact = S.cosφ + im * S.sinφ

	for m in 1:L
		sig *= -1
		ep *= ep_fact            # ep =   exp(i *   m  * φ)
		em = sig * conj(ep)      # ep = ± exp(i * (-m) * φ)
		dep_dφ = im *   m  * ep
		dem_dφ = im * (-m) * em
		for l in m:L
			p_div_sinθ = P[index_p(l,m)]
			@inbounds Y[index_y(l, -m)] = em * p_div_sinθ * S.sinθ
			@inbounds Y[index_y(l,  m)] = ep * p_div_sinθ * S.sinθ

			dp_dθ = dP[index_p(l,m)]
			@inbounds dY[index_y(l, -m)] = dspher_to_dcart(S, dem_dφ * p_div_sinθ, em * dp_dθ)
			@inbounds dY[index_y(l,  m)] = dspher_to_dcart(S, dep_dφ * p_div_sinθ, ep * dp_dθ)
		end
	end
	# return Y, dY
end



# revive if needed
# """
# 	cYlm_from_xz(L, x, z)
#
# Compute an entire set of real spherical harmonics ``Y_{l,m}(θ, φ)`` for
# ``x = cos θ, z = sin φ`` where ``0 ≤ l ≤ L`` and ``-l ≤ m ≤ l``.
# """
# function cYlm_from_cart(L::Integer, R::JVec{T}) where {T}
# 	S = cart2spher(R)
# 	P = Vector{T}(undef, sizeP(L))
# 	coeff = compute_coefficients(L)
# 	compute_p!(L, S, coeff, P)
# 	Y = Vector{ComplexF64}(undef, sizeY(L))
# 	cYlm!(Y, L, S, P)
# 	return Y
# end

# ---------------------------------------------
#      Nicer interface
# ---------------------------------------------

struct SHBasis{T}
	maxL::Int
	coeff::ALPCoefficients{T}
end

import Base.==
==(B1::SHBasis, B2::SHBasis) = (B1.maxL == B1.maxL)

SHBasis(maxL::Integer, T=Float64) =
		SHBasis(maxL, compute_coefficients(maxL))

Base.eltype(SH::SHBasis{T}) where {T} = T
Base.length(S::SHBasis) = sizeY(S.maxL)

alloc_B( S::SHBasis{T}) where {T} =
		Vector{Complex{T}}(undef, length(S))

alloc_dB(S::SHBasis{T}, args...) where {T} =
		Vector{JVec{Complex{T}}}(undef, length(S))

alloc_temp(SH::SHBasis{T}, args...) where {T} = (
		P = Vector{T}(undef, sizeP(SH.maxL)), )

alloc_temp_d(SH::SHBasis{T}, args...) where {T} = (
		 P = Vector{T}(undef, sizeP(SH.maxL)),
		dP = Vector{T}(undef, sizeP(SH.maxL)) )


function eval_basis!(Y, SH::SHBasis, R::JVec, tmp)
	L=SH.maxL
	@assert 0 <= L <= SH.maxL
	@assert length(Y) >= sizeY(L)
	S = cart2spher(R)
	compute_p!(L, S, SH.coeff, tmp.P)
	cYlm!(Y, L, S, tmp.P)
	return Y
end


function eval_basis_d!(Y, dY, SH::SHBasis, R::JVec, tmp)
	L=SH.maxL
	@assert 0 <= L <= SH.maxL
	@assert length(Y) >= sizeY(L)
	# if R[1]^2+R[2]^2 < 1e-20 * R[3]^2
	# 	R = JVec(R[1]+1e-9, R[2], R[3])
	# end
	S = cart2spher(R)
	compute_dp!(L, S, SH.coeff, tmp.P, tmp.dP)
	cYlm_d!(Y, dY, L, S, tmp.P, tmp.dP)
	# return Y, dY
end


# ---------------- Clebsch Gordan Stuff


export clebschgordan, cg1

cg_l_condition(j1, j2, j3) = (abs(j1-j2) <= j3 <= j1 + j2)
cg_m_condition(m1, m2, m3) = (m3 == m1 + m2)

"""
`cg1(j1, m1, j2, m2, j3, m3, T=Float64)` : A reference implementation of
Clebsch-Gordon coefficients based on

https://hal.inria.fr/hal-01851097/document
Equation (4-6)

This heavily uses BigInt and BigFloat and should therefore not be employed
for performance critical tasks.
"""
function cg1(j1, m1, j2, m2, j3, m3, T=Float64)
   if !cg_m_condition(m1, m2, m3) || !cg_l_condition(j1, j2, j3)
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

clebschgordan = cg1

# TODO: reduce dimensionality of the storage tensor
#       to numY x numY x maxL (or possibly less?)
struct ClebschGordan{T}
	maxL::Int
	cg::Array{T, 3}
end

function ClebschGordan(maxL, T=Float64)
	n = sizeY(maxL)
	cg = zeros(T, n, n, n)
	# TODO: insert restrictions on (j1,j2,j3)-values!
	for j1 = 0:maxL, j2 = 0:maxL, j3 = 0:maxL
		if !cg_l_condition(j1, j2, j3)
			continue
		end
		for m1 = -j1:j1, m2 = -j2:j2
			m3 = m1 + m2  # cf. cg_m_condition
			if abs(m3) > j3
				continue
			end
			cg[index_y(j1,m1), index_y(j2,m2), index_y(j3,m3)] =
					clebschgordan(j1,m1,j2,m2,j3,m3)
		end
	end
	return ClebschGordan(maxL, cg)
end

(cg::ClebschGordan)(j1,m1,j2,m2,j3,m3) =
	cg.cg[index_y(j1,m1), index_y(j2,m2), index_y(j3,m3)]

end
