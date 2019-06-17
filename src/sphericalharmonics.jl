
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module SphericalHarmonics


import SHIPs
using StaticArrays, LinearAlgebra

const SVec3 = SVector{3}

export SHBasis


struct PseudoSpherical{T}
	r::T
	cosφ::T
	sinφ::T
	cosθ::T
	sinθ::T
end

function cart2spher(R::SVec3)
	r = norm(R)
	φ = atan(R[2], R[1])
	sinφ, cosφ = sincos(φ)
	cosθ = R[3] / r
	sinθ = sqrt(1-cosθ^2)
	return PseudoSpherical(r, cosφ, sinφ, cosθ, sinθ)
end


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
index_p(l,m) = m + div(l*(l+1), 2) + 1

"""
	index_y(l,m)

Return the index into a flat array of real spherical harmonics ``Y_{l,m}``
for the given indices ``(l,m)``.
``Y_{l,m}`` are stored in l-major order i.e.
[Y(0,0), [Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), ...]
"""
index_y(l,m) = m + l + (l*l) + 1


# --------------------------------------------------------
#     Associated Legendre Polynomials
#     TODO: rewrite within general interface?
#           - alloc_B, alloc_dB, alloc_temp, ...
# --------------------------------------------------------

"""
TODO: documentation
"""
struct ALPCoefficients
	A::Array{Float64}
	B::Array{Float64}
end

ALPCoefficients(maxDegree::Int) =
	ALPCoefficients( Array{Float64}(undef, sizeP(maxDegree)),
						  Array{Float64}(undef, sizeP(maxDegree)) )

"""
	compute_coefficients(L)

Precompute coefficients ``a_l^m`` and ``b_l^m`` for all l <= L, m <= l
"""
function compute_coefficients(L::Int)
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
function compute_p!(L::Int, S::PseudoSpherical, coeff::ALPCoefficients,
					     P::Array{Float64,1})
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
			for m in 0:(l-2)
				P[index_p(l, m)] =
						coeff.A[index_p(l, m)] * (     S.cosθ * P[index_p(l - 1, m)]
						             + coeff.B[index_p(l, m)] * P[index_p(l - 2, m)] )
			end
			P[index_p(l, l - 1)] = S.cosθ * sqrt(2 * (l - 1) + 3) * temp
			temp = -sqrt(1.0 + 0.5 / l) * S.sinθ * temp
			P[index_p(l, l)] = temp
		end
	end
	return P
end

"""
dP = dP / dθ (and not dP / dx!!!)
"""
function compute_dp!(L::Int, S::PseudoSpherical, coeff::ALPCoefficients,
					     P::Array{Float64,1}, dP::Array{Float64,1})
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

		temp, temp_d = ( - sqrt(1.5) * S.sinθ * temp,
						     - sqrt(1.5) * (S.cosθ * temp + S.sinθ * temp_d) )
		P[index_p(1, 1)] = temp
		dP[index_p(1, 1)] = temp_d

		for l in 2:L
			for m in 0:(l-2)
				P[index_p(l, m)] =
						coeff.A[index_p(l, m)] * (     S.cosθ * P[index_p(l - 1, m)]
						             + coeff.B[index_p(l, m)] * P[index_p(l - 2, m)] )
				dP[index_p(l, m)] =
					coeff.A[index_p(l, m)] * (
									- S.sinθ * P[index_p(l - 1, m)]
									+ S.cosθ * dP[index_p(l - 1, m)]
					             + coeff.B[index_p(l, m)] * dP[index_p(l - 2, m)] )
			end
			P[index_p(l, l - 1)] = sqrt(2 * (l - 1) + 3) * S.cosθ * temp
			dP[index_p(l, l - 1)] = sqrt(2 * (l - 1) + 3) * (
										        -S.sinθ * temp + S.cosθ * temp_d )

         (temp, temp_d) = (-sqrt(1.0+0.5/l) * S.sinθ * temp,
						         -sqrt(1.0+0.5/l) * (S.cosθ * temp + S.sinθ * temp_d) )
			P[index_p(l, l)] = temp
			dP[index_p(l, l)] = temp_d
		end
	end
	return P, dP
end



# function compute_dp!(L::Int, S::PseudoSpherical, coeff::ALPCoefficients,
# 					      P, dP)
#    @assert L > 0
# 	@assert length(coeff.A) >= sizeP(L)
# 	@assert length(coeff.B) >= sizeP(L)
# 	@assert length(P) >= sizeP(L)
# 	@assert length(dP) >= sizeP(L)
#
# 	x = S.cosθ
# 	sinθ = S.sinθ
# 	sinθ_dθ = x
# 	x_dθ = - sinθ
#
# 	temp = 0.39894228040143267794 # = sqrt(0.5/M_PI)
# 	P[index_p(0, 0)] = temp
# 	dP[index_p(0, 0)] = 0
#
# 	SQRT3 = 1.7320508075688772935
# 	P[index_p(1, 0)] = x * SQRT3 * temp
# 	dP[index_p(1, 0)] = x_dθ * SQRT3 * temp
#
# 	SQRT3DIV2 = -1.2247448713915890491
# 	temp = SQRT3DIV2 * sinθ * temp
# 	temp_dθ = SQRT3DIV2 * sinθ_dθ * temp
# 	P[index_p(1, 1)] = temp
# 	dP[index_p(1, 1)] = temp_dθ
#
# 	for l in 2:L
# 		for m in 0:(l-2)
# 			P[index_p(l, m)] =
# 					coeff.A[index_p(l, m)] * (
# 						x * P[index_p(l - 1, m)]
# 					     + coeff.B[index_p(l, m)] * P[index_p(l - 2, m)]
# 				   )
# 			dP[index_p(l, m)] =
# 					coeff.A[index_p(l, m)] * (
# 						x_dθ * P[index_p(l - 1, m)]
# 						+ x * dP[index_p(l - 1, m)]
# 					   + coeff.B[index_p(l, m)] * dP[index_p(l - 2, m)]
# 				   )
# 		end
# 		P[index_p(l, l - 1)] = x * sqrt(2 * (l - 1) + 3) * temp
# 		dP[index_p(l, l - 1)] = ( x_dθ * sqrt(2 * (l - 1) + 3) * temp
# 		     							  + x * sqrt(2 * (l - 1) + 3) * temp_dθ )
# 		temp = -sqrt(1.0 + 0.5 / l) * sinθ * temp
# 		temp_dθ = ( -sqrt(1.0 + 0.5 / l) * sinθ_dθ * temp
# 		            -sqrt(1.0 + 0.5 / l) * sinθ * temp_dθ )
# 		P[index_p(l, l)] = temp
# 		dP[index_p(l, l)] = temp_dθ
# 	end
# 	return P, dP
# end


"""
	compute_p(L, x)

Compute an entire set of Associated Legendre Polynomials ``P_l^m(x)`` where
``0 ≤ l ≤ L`` and ``0 ≤ m ≤ l``. Assumes ``|x| ≤ 1``.
"""
function compute_p(L::Integer, S::PseudoSpherical)
	P = Array{Float64}(undef, sizeP(L))
	coeff = compute_coefficients(L)
	compute_p!(L, S, coeff, P)
	return P
end

function compute_dp(L::Integer, S::PseudoSpherical)
	P = Array{Float64}(undef, sizeP(L))
	dP = Array{Float64}(undef, sizeP(L))
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


# OLD VERSION: somewhat faster -> return to this if it is a bottleneck
# """
# R = r(cosφ sinθ, sinφ sinθ, cosθ)
# x = sinφ
# z = cosθ
# s picks the correct inverse of sinφ -> φ
# """
# function compute_rxz(R::SVec3{T}) where {T}
#    r = norm(R)
#    z = R[3] / r
#    x = R[2] / sqrt(1 - z^2) / r
#    s = sign(R[1])
#    return r, x, z, s
# end

function cYlm!(Y, L, S::PseudoSpherical, P)
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
   @assert abs(S.cosθ) <= 1.0

	INVSQRT2 = 1 / sqrt(2)

	for l = 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * INVSQRT2
	end

   sig = 1
   ep = INVSQRT2
   ep_fact = S.cosφ + im * S.sinφ
	for m in 1:L
		sig *= -1
		ep *= ep_fact            # ep =   exp(i *   m  * φ)
		em = sig * conj(ep)      # ep = ± exp(i * (-m) * φ)
		for l in m:L
			p = P[index_p(l,m)]
			Y[index_y(l, -m)] = em * p   # (-1)^m * p * exp(-im*m*phi) / sqrt(2)
			Y[index_y(l,  m)] = ep * p   #          p * exp( im*m*phi) / sqrt(2)
		end
	end

	return Y
end

"""
convert a gradient with respect to spherical coordinates to a gradient
with respect to cartesian coordinates
"""
dspher_to_dcart(S, f_φ, f_θ) =
	SVector( S.sinφ * S.sinθ * f_φ + S.cosθ * f_θ,
	         (S.cosφ * f_φ / S.r) / S.sinθ,
				(S.cosθ * S.sinφ / S.r) * f_φ - (S.sinθ / S.r) * f_θ )

function cYlm_d!(Y, dY, L, S::PseudoSpherical, P, dP)
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
   @assert abs(S.cosθ) <= 1.0

	INVSQRT2 = 1 / sqrt(2)

	for l = 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * INVSQRT2
		dY[index_y(l, 0)] = dspher_to_dcart(S, 0.0, dP[index_p(l, 0)] * INVSQRT2)
	end

   sig = 1
   ep = INVSQRT2
   ep_fact = S.cosφ + im * S.sinφ

	for m in 1:L
		sig *= -1
		ep *= ep_fact            # ep =   exp(i *   m  * φ)
		em = sig * conj(ep)      # ep = ± exp(i * (-m) * φ)
		ep_dφ = im * m * ep
		em_dφ = - im * m * em

		for l in m:L
			p = P[index_p(l,m)]
			Y[index_y(l, -m)] = em * p   # (-1)^m * p * exp(-im*m*phi) / sqrt(2)
			Y[index_y(l,  m)] = ep * p   #          p * exp( im*m*phi) / sqrt(2)

			p_dθ = dP[index_p(l,m)]
			dY[index_y(l, -m)] = dspher_to_dcart(S, em_dφ * p, em * p_dθ)
			dY[index_y(l,  m)] = dspher_to_dcart(S, ep_dφ * p, ep * p_dθ)
		end
	end

	return Y, dY
end


# revive if needed
# """
# 	cYlm_from_xz(L, x, z)
#
# Compute an entire set of real spherical harmonics ``Y_{l,m}(θ, φ)`` for
# ``x = cos θ, z = sin φ`` where ``0 ≤ l ≤ L`` and ``-l ≤ m ≤ l``.
# """
# function cYlm_from_cart(L::Integer, R::SVec3{T}) where {T}
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
	P::Vector{T}
	dP::Vector{T}
	coeff::ALPCoefficients
end

SHBasis(maxL::Integer, T=Float64) =
		SHBasis(maxL, Vector{T}(undef, sizeP(maxL)),
					     Vector{T}(undef, sizeP(maxL)),
						  compute_coefficients(maxL))

Base.length(S::SHBasis) = sizeY(S.maxL)

SHIPs.alloc_B( S::SHBasis{T}) where {T} =
		Vector{Complex{T}}(undef, length(S))
SHIPs.alloc_dB(S::SHBasis{T}) where {T} =
		Vector{SVec3{Complex{T}}}(undef, length(S))

function SHIPs.eval_basis!(Y, SH::SHBasis, R::SVec3, L=SH.maxL)
	@assert 0 <= L <= SH.maxL
	@assert length(Y) >= sizeY(L)
	S = cart2spher(R)
	compute_p!(L, S, SH.coeff, SH.P)
	cYlm!(Y, L, S, SH.P)
	return Y
end


function SHIPs.eval_basis_d!(Y, dY, SH::SHBasis, R::SVec3, L=SH.maxL)
	@assert 0 <= L <= SH.maxL
	@assert length(Y) >= sizeY(L)
	S = cart2spher(R)
	compute_dp!(L, S, SH.coeff, SH.P, SH.dP)
	cYlm_d!(Y, dY, L, S, SH.P, SH.dP)
	return Y, dY
end


# ---------------- Clebsch Gordan Stuff


export clebschgordan, cg1



"""
`cg1(j1, m1, j2, m2, j3, m3, T=Float64)` : A reference implementation of
Clebsch-Gordon coefficients based on

https://hal.inria.fr/hal-01851097/document
Equation (4-6)

This heavily uses BigInt and BigFloat and should therefore not be employed
for performance critical tasks.
"""
function cg1(j1, m1, j2, m2, j3, m3, T=Float64)
   if (m3 != m1 + m2) || !(abs(j1-j2) <= j3 <= j1 + j2)
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

end
