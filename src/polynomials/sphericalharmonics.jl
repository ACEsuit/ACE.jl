
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



module SphericalHarmonics

using StaticArrays, LinearAlgebra

import ACE

import JuLIP.MLIPs: IPBasis, alloc_B, alloc_dB, fltype
import JuLIP: alloc_temp, alloc_temp_d, evaluate!, evaluate_d!,
			     write_dict, read_dict

const JVec = SVector{3}

export SHBasis, RSHBasis



# --------------------------------------------------------
#     Coordinates
# --------------------------------------------------------


"""
`struct PseudoSpherical` : a simple datatype storing spherical coordinates
of a point (x,y,z) in the format (r, cosφ, sinφ, cosθ, sinθ).

Use `spher2cart` and `cart2spher` to convert between cartesian and spherical
coordinates.
"""
struct PseudoSpherical{T}
	r::T
	cosφ::T
	sinφ::T
	cosθ::T
	sinθ::T
end

spher2cart(S::PseudoSpherical) = S.r * JVec(S.cosφ*S.sinθ, S.sinφ*S.sinθ, S.cosθ)

function cart2spher(R::AbstractVector)
	@assert length(R) == 3
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
@inline index_p(l::Int,m::Int) = m + div(l*(l+1), 2) + 1
index_p(l::Integer, m::Integer) = index_y(Int(l), Int(m))

"""
	index_y(l,m)

Return the index into a flat array of real spherical harmonics ``Y_{l,m}``
for the given indices ``(l,m)``.
``Y_{l,m}`` are stored in l-major order i.e.
[Y(0,0), [Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), ...]
"""
Base.@pure index_y(l::Int, m::Int) = m + l + (l*l) + 1
index_y(l::Integer, m::Integer) = index_y(Int(l), Int(m))

# # i = m + l + l^2 + 1
# # m = i - l - l^2 - 1
# _li2m(l, i) = i - l - l^2 - 1

# --------------------------------------------------------
#     Associated Legendre Polynomials
#     TODO: rewrite within general interface?
#           - alloc_B, alloc_dB, alloc_temp, ...
# --------------------------------------------------------

"""
`ALPCoefficients` : an auxiliary datastructure for
evaluating the associated lagrange functions
used for the spherical harmonics
"""
struct ALPCoefficients{T}
	A::Vector{T}
	B::Vector{T}
end

ALPCoefficients(maxDegree::Integer, T::Type=Float64) =
	ALPCoefficients( Vector{T}(undef, sizeP(maxDegree)),
						  Vector{T}(undef, sizeP(maxDegree)) )

"""
	compute_coefficients(L)

Precompute coefficients ``a_l^m`` and ``b_l^m`` for all l <= L, m <= l
"""
function compute_coefficients(L::Integer, T::Type=Float64)
	coeff = ALPCoefficients(L, T)
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
	@assert length(coeff.A) >= sizeP(L)
	@assert length(coeff.B) >= sizeP(L)
	@assert length(P) >= sizeP(L)

	temp = sqrt(0.5/π)
	P[index_p(0, 0)] = temp
	if L == 0; return P; end

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

	return P
end


function compute_dp!(L::Integer, S::PseudoSpherical{T}, coeff::ALPCoefficients{T},
					     P::Array{T,1}, dP::Array{T,1}) where T
	@assert length(coeff.A) >= sizeP(L)
	@assert length(coeff.B) >= sizeP(L)
	@assert length(P) >= sizeP(L)

	temp = sqrt(0.5/π)
	P[index_p(0, 0)] = temp
	temp_d = 0.0
	dP[index_p(0, 0)] = temp_d
	if L == 0; return nothing; end

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
	return nothing
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

"""
evaluate complex spherical harmonics
"""
function cYlm!(Y, L, S::PseudoSpherical, P)
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
   @assert abs(S.cosθ) <= 1.0

	ep = 1 / sqrt(2) + im * 0
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


"""
evaluate gradients of complex spherical harmonics
"""
function cYlm_d!(Y, dY, L, S::PseudoSpherical, P, dP)
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
	@assert length(dY) >= sizeY(L)
   # @assert abs(S.cosθ) < 1.0

	# m = 0 case
	ep = 1 / sqrt(2)
	for l = 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * ep
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


"""
evaluate real spherical harmonics
"""
function rYlm!(Y::AbstractVector{T}, L, S::PseudoSpherical, P) where {T <: Real}
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
   @assert abs(S.cosθ) <= 1.0

   oort2 = 1 / sqrt(2)
	for l = 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * oort2
	end

   sig = 1
	ec = 1.0 + 0 * im
   ec_fact = S.cosφ + im * S.sinφ
	for m in 1:L
		sig *= -1                # sig = (-1)^m
		ec *= ec_fact            # ec = exp(i * m  * φ) / sqrt(2)
		# cYlm = p * ec,    (also cYl{-m} = sig * p * conj(ec)), but not needed)
		# rYlm    =  Re(cYlm)
		# rYl{-m} = -Im(cYlm)
		for l in m:L
			p = P[index_p(l,m)]
			@inbounds Y[index_y(l, -m)] = -p * imag(ec)
			@inbounds Y[index_y(l,  m)] =  p * real(ec)
		end
	end

	return Y
end


"""
evaluate gradients of real spherical harmonics
"""
function rYlm_d!(Y::AbstractVector{T}, dY,
					  L, S::PseudoSpherical, P, dP) where {T <: Real}
	@assert length(P) >= sizeP(L)
	@assert length(Y) >= sizeY(L)
   @assert abs(S.cosθ) <= 1.0

   oort2 = 1 / sqrt(2)
	for l = 0:L
		Y[index_y(l, 0)] = P[index_p(l, 0)] * oort2
		dY[index_y(l, 0)] = dspher_to_dcart(S, 0.0, dP[index_p(l, 0)] * oort2)
	end

   sig = 1
	ec = 1.0 + 0 * im
   ec_fact = S.cosφ + im * S.sinφ
	drec_dφ = 0.0
	for m in 1:L
		sig *= -1                # sig = (-1)^m
		ec *= ec_fact            # ec = exp(i * m  * φ) / sqrt(2)
		dec_dφ = im * m * ec

		# cYlm = p * ec,    (also cYl{-m} = sig * p * conj(ec)), but not needed)
		# rYlm    =  Re(cYlm)
		# rYl{-m} = -Im(cYlm)
		for l in m:L
			p_div_sinθ = P[index_p(l,m)]
			p = p_div_sinθ * S.sinθ
			Y[index_y(l, -m)] = -p * imag(ec)
			Y[index_y(l,  m)] =  p * real(ec)

			dp_dθ = dP[index_p(l,m)]
			dY[index_y(l, -m)] = dspher_to_dcart(S, - imag(dec_dφ) * p_div_sinθ,
															    - imag(ec) * dp_dθ)
			dY[index_y(l,  m)] = dspher_to_dcart(S,   real(dec_dφ) * p_div_sinθ,
															      real(ec) * dp_dθ)
		end
	end

	return dY
end


# ---------------------------------------------
#      The nice basis interface
# ---------------------------------------------

abstract type AbstractSHBasis{T} <: IPBasis end

"""
complex spherical harmonics
"""
struct SHBasis{T} <: AbstractSHBasis{T}
	maxL::Int
	coeff::ALPCoefficients{T}
end

"""
real spherical harmonics
"""
struct RSHBasis{T} <: AbstractSHBasis{T}
	maxL::Int
	coeff::ALPCoefficients{T}
end


import Base.==
==(B1::AbstractSHBasis, B2::AbstractSHBasis) =
	( (B1.maxL == B1.maxL) &&
	  (typeof(B1) == typeof(B2)) )

write_dict(SH::SHBasis{T}) where {T} =
		Dict("__id__" => "ACE_SHBasis",
			  "T" => write_dict(T),
			  "maxL" => SH.maxL)

read_dict(::Val{:SHIPs_SHBasis}, D::Dict) =
	read_dict(Val{:ACE_SHBasis}(), D)

read_dict(::Val{:ACE_SHBasis}, D::Dict) =
		SHBasis(D["maxL"], read_dict(D["T"]))

SHBasis(maxL::Integer, T::Type=Float64) =
		SHBasis(Int(maxL), compute_coefficients(maxL, T))

RSHBasis(maxL::Integer, T::Type=Float64) =
		RSHBasis(Int(maxL), compute_coefficients(maxL, T))

rfltype(SH::AbstractSHBasis{T}) where {T} = T
Base.length(S::AbstractSHBasis) = sizeY(S.maxL)

fltype(::SHBasis{T}) where {T} = Complex{T}
fltype(::RSHBasis{T}) where {T} = T

alloc_B( S::SHBasis{T}, args...) where {T} =
		Vector{Complex{T}}(undef, length(S))

alloc_B( S::RSHBasis{T}, args...) where {T} =
		Vector{T}(undef, length(S))

alloc_dB(S::SHBasis{T}) where {T} =
		Vector{JVec{Complex{T}}}(undef, length(S))

alloc_dB(S::RSHBasis{T}) where {T} =
		Vector{JVec{T}}(undef, length(S))

alloc_dB(S::AbstractSHBasis, N::Integer) = alloc_dB(S)

alloc_temp(SH::AbstractSHBasis{T}, args...) where {T} = (
		P = Vector{T}(undef, sizeP(SH.maxL)), )

alloc_temp_d(SH::AbstractSHBasis{T}, args...) where {T} = (
		 P = Vector{T}(undef, sizeP(SH.maxL)),
		dP = Vector{T}(undef, sizeP(SH.maxL)) )


_evaluate!(Y, L, S, P, ::SHBasis) = cYlm!(Y, L, S, P)
_evaluate!(Y, L, S, P, ::RSHBasis) = rYlm!(Y, L, S, P)

_evaluate_d!(Y, dY, L, S, P, dP, ::SHBasis) = cYlm_d!(Y, dY, L, S, P, dP)
_evaluate_d!(Y, dY, L, S, P, dP, ::RSHBasis) = rYlm_d!(Y, dY, L, S, P, dP)

function evaluate!(Y, tmp, SH::AbstractSHBasis, R::JVec)
	L=SH.maxL
	@assert 0 <= L <= SH.maxL
	@assert length(Y) >= sizeY(L)
	S = cart2spher(R)
	compute_p!(L, S, SH.coeff, tmp.P)
	_evaluate!(Y, L, S, tmp.P, SH)
	return Y
end


function evaluate_d!(Y, dY, tmp, SH::AbstractSHBasis, R::JVec)
	L=SH.maxL
	@assert 0 <= L <= SH.maxL
	@assert length(Y) >= sizeY(L)
	# if R[1]^2+R[2]^2 < 1e-20 * R[3]^2
	# 	R = JVec(R[1]+1e-9, R[2], R[3])
	# end
	S = cart2spher(R)
	compute_dp!(L, S, SH.coeff, tmp.P, tmp.dP)
	_evaluate_d!(Y, dY, L, S, tmp.P, tmp.dP, SH)
	# return Y, dY
end


end
