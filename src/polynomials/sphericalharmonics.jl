


module SphericalHarmonics

using StaticArrays, LinearAlgebra

import ACE

import ACE: alloc_B, alloc_dB, fltype, valtype, grad_type, 
		      alloc_temp, alloc_temp_d, evaluate!, evaluate_d!, evaluate_ed!,
			   write_dict, read_dict,
				ACEBasis

export SHBasis


# --------------------------------------------------------
#     Coordinates
# --------------------------------------------------------


"""
`struct SphericalCoords` : a simple datatype storing spherical coordinates
of a point (x,y,z) in the format `(r, cosφ, sinφ, cosθ, sinθ)`.

Use `spher2cart` and `cart2spher` to convert between cartesian and spherical
coordinates.
"""
struct SphericalCoords{T}
	r::T
	cosφ::T
	sinφ::T
	cosθ::T
	sinθ::T
end

spher2cart(S::SphericalCoords) = S.r * SVector(S.cosφ*S.sinθ, S.sinφ*S.sinθ, S.cosθ)

function cart2spher(R::AbstractVector)
	@assert length(R) == 3
	r = norm(R)
	φ = atan(R[2], R[1])
	sinφ, cosφ = sincos(φ)
	cosθ = R[3] / r
	sinθ = sqrt(R[1]^2+R[2]^2) / r
	return SphericalCoords(r, cosφ, sinφ, cosθ, sinθ)
end

SphericalCoords(φ, θ) = SphericalCoords(1.0, cos(φ), sin(φ), cos(θ), sin(θ))
SphericalCoords(r, φ, θ) = SphericalCoords(r, cos(φ), sin(φ), cos(θ), sin(θ))

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
`sizeP(maxL):` 
Return the size of the set of Associated Legendre Polynomials ``P_l^m(x)`` of
degree less than or equal to the given maximum degree
"""
sizeP(maxL) = div((maxL + 1) * (maxL + 2), 2)

"""
`sizeY(maxL):`
Return the size of the set of real spherical harmonics ``Y_{l,m}(θ,φ)`` of
degree less than or equal to the given maximum degree
"""
sizeY(maxL) = (maxL + 1) * (maxL + 1)

"""
`index_p(l,m):`
Return the index into a flat array of Associated Legendre Polynomials `P_l^m`
for the given indices `(l,m)`.
`P_l^m` are stored in l-major order i.e. `[P(0,0), [P(1,0), P(1,1), P(2,0), ...]``
"""
index_p(l::Integer,m::Integer) = m + div(l*(l+1), 2) + 1

"""
`index_y(l,m):`
Return the index into a flat array of real spherical harmonics `Y_lm`
for the given indices `(l,m)`.
`Y_lm` are stored in l-major order i.e.
[Y(0,0), Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), ...]
"""
index_y(l::Integer, m::Integer) = m + l + (l*l) + 1



# --------------------------------------------------------
#     Associated Legendre Polynomials
#     TODO: possibly rewrite within general interface?
#           - alloc_B, alloc_dB, alloc_temp, ...
# --------------------------------------------------------

"""
`ALPolynomials` : an auxiliary datastructure for
evaluating the associated lagrange functions
used for the spherical harmonics

Constructor:
```julia
ALPolynomials(maxL::Integer, T::Type=Float64)
```
"""
struct ALPolynomials{T} <: ACEBasis
	L::Int
	A::Vector{T}
	B::Vector{T}
end

fltype(alp::ALPolynomials{T}) where {T} = T 

Base.length(alp::ALPolynomials) = sizeP(alp.L)

function ALPolynomials(L::Integer, T::Type=Float64)
	# Precompute coefficients ``a_l^m`` and ``b_l^m`` for all l <= L, m <= l
	alp = ALPolynomials(L, Vector{T}(undef, sizeP(L)), Vector{T}(undef, sizeP(L)))
	for l in 2:L
		ls = l*l
		lm1s = (l-1) * (l-1)
		for m in 0:(l-2)
			ms = m * m
			alp.A[index_p(l, m)] = sqrt((4 * ls - 1.0) / (ls - ms))
			alp.B[index_p(l, m)] = -sqrt((lm1s - ms) / (4 * lm1s - 1.0))
		end
	end
	return alp
end

# _alp_pool = 
# release_B!(alp::ALPolynomials, B) = release!(_alp_pool, B)

alloc_B(alp::ALPolynomials) = Vector{Float64}(undef, sizeP(alp.L))

alloc_dB(alp::ALPolynomials, args...) = Vector{Float64}(undef, sizeP(alp.L))

function evaluate!(P, tmp, alp::ALPolynomials, S::SphericalCoords)
	L = alp.L 
	A = alp.A 
	B = alp.B 
	@assert length(A) >= sizeP(L)
	@assert length(B) >= sizeP(L)
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
			@inbounds P[il+m] = A[il+m] * (     S.cosθ * P[ilm1+m]
  					                           + B[il+m] * P[ilm2+m] )
		end
		@inbounds P[il+l-1] = S.cosθ * sqrt(2 * (l - 1) + 3) * temp
		temp = -sqrt(1.0 + 0.5 / l) * S.sinθ * temp
		@inbounds P[il+l] = temp
	end

	return P
end


_evaluate_ed(alp::ALPolynomials, S::SphericalCoords) = 
	_evaluate_ed!(alloc_B(alp), alloc_dB(alp), alp::ALPolynomials, S::SphericalCoords)

# this doesn't use the standard name because it doesn't 
# technically perform the derivative w.r.t. S, but w.r.t. θ
# further, P doesn't store P but (P if m = 0) or (P * sinθ if m > 0)
# this is done for numerical stability 
function _evaluate_ed!(P, dP, alp::ALPolynomials, S::SphericalCoords)
	L = alp.L 
	A = alp.A 
	B = alp.B 
	@assert length(A) >= sizeP(L)
	@assert length(B) >= sizeP(L)
	@assert length(P) >= sizeP(L)
	@assert length(dP) >= sizeP(L)

	temp = sqrt(0.5/π)
	P[index_p(0, 0)] = temp
	temp_d = 0.0
	dP[index_p(0, 0)] = temp_d
	if L == 0; return P, dP; end

	P[index_p(1, 0)] = S.cosθ * sqrt(3) * temp
	dP[index_p(1, 0)] = -S.sinθ * sqrt(3) * temp + S.cosθ * sqrt(3) * temp_d
	temp1, temp_d = ( - sqrt(1.5) * temp,
					      - sqrt(1.5) * (S.cosθ * temp + S.sinθ * temp_d) )
	P[index_p(1, 1)] = temp1
	dP[index_p(1, 1)] = temp_d

	for l in 2:L
		m = 0
		@inbounds P[index_p(l, m)] =
				A[index_p(l, m)] * (     S.cosθ * P[index_p(l - 1, m)]
				             + B[index_p(l, m)] * P[index_p(l - 2, m)] )
		@inbounds dP[index_p(l, m)] =
			A[index_p(l, m)] * (
							- S.sinθ * P[index_p(l - 1, m)]
							+ S.cosθ * dP[index_p(l - 1, m)]
			             + B[index_p(l, m)] * dP[index_p(l - 2, m)] )

		for m in 1:(l-2)
			@inbounds P[index_p(l, m)] =
					A[index_p(l, m)] * (     S.cosθ * P[index_p(l - 1, m)]
					             + B[index_p(l, m)] * P[index_p(l - 2, m)] )
			@inbounds dP[index_p(l, m)] =
				A[index_p(l, m)] * (
								- S.sinθ^2 * P[index_p(l - 1, m)]
								+ S.cosθ * dP[index_p(l - 1, m)]
				             + B[index_p(l, m)] * dP[index_p(l - 2, m)] )
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

	return P, dP
end




# ------------------------------------------------------------------------
#                  Spherical Harmonics
# ------------------------------------------------------------------------

"""
evaluate complex spherical harmonics
"""
function cYlm!(Y, L, S::SphericalCoords, P)
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
function cYlm_d!(dY, L, S::SphericalCoords, P, dP)
	@assert length(P) >= sizeP(L)
	@assert length(dY) >= sizeY(L)
   # @assert abs(S.cosθ) < 1.0

	# m = 0 case
	ep = 1 / sqrt(2)
	for l = 0:L
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
			dp_dθ = dP[index_p(l,m)]
			@inbounds dY[index_y(l, -m)] = dspher_to_dcart(S, dem_dφ * p_div_sinθ, em * dp_dθ)
			@inbounds dY[index_y(l,  m)] = dspher_to_dcart(S, dep_dφ * p_div_sinθ, ep * dp_dθ)
		end
	end
	return dY
end


"""
evaluate gradients of complex spherical harmonics
"""
function cYlm_ed!(Y, dY, L, S::SphericalCoords, P, dP)
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





# ---------------------------------------------
#      The nice basis interface
# ---------------------------------------------

"""
This extra abstraction is no longer needed, but there use to be a real SH basis
and in case this is revived, I am keeping it for now. 
"""
abstract type AbstractSHBasis{T} <: ACEBasis end

"""
complex spherical harmonics
"""
struct SHBasis{T} <: AbstractSHBasis{T}
	alp::ALPolynomials{T}
end

maxL(sh::AbstractSHBasis) = sh.alp.L



import Base.==
==(B1::AbstractSHBasis, B2::AbstractSHBasis) =
		(B1.alp == B2.alp) && (typeof(B1) == typeof(B2))

write_dict(SH::SHBasis{T}) where {T} =
		Dict("__id__" => "ACE_SHBasis",
			  "T" => write_dict(T),
			  "maxL" => maxL(SH))

read_dict(::Val{:ACE_SHBasis}, D::Dict) =
		SHBasis(D["maxL"], read_dict(D["T"]))

SHBasis(maxL::Integer, T::Type=Float64) =
		SHBasis(ALPolynomials(maxL, T))

Base.length(S::AbstractSHBasis) = sizeY(maxL(S))

ACE.valtype(::SHBasis{T}, args...) where {T} = Complex{T}

gradtype(::SHBasis{T}) where {T} = SVector{3, Complex{T}}

alloc_B( S::SHBasis{T}, args...) where {T} =
		Vector{Complex{T}}(undef, length(S))

alloc_dB(S::SHBasis{T}, args...) where {T} =
		Vector{SVector{3, Complex{T}}}(undef, length(S))

alloc_temp(SH::AbstractSHBasis{T}, args...) where {T} = (
		P = alloc_B(SH.alp), )

alloc_temp_d(SH::AbstractSHBasis{T}, args...) where {T} = (
		 P = alloc_B(SH.alp),
		dP = alloc_dB(SH.alp) )

const P_pool = ACE.ObjectPools.StaticVectorPool{Float64}()
const dP_pool = ACE.ObjectPools.StaticVectorPool{Float64}()
		


_evaluate!(Y, L, S, P, ::SHBasis) = cYlm!(Y, L, S, P)

_evaluate_d!(dY, L, S, P, dP, ::SHBasis) = cYlm_d!(dY, L, S, P, dP)

_evaluate_ed!(Y, dY, L, S, P, dP, ::SHBasis) = cYlm_ed!(Y, dY, L, S, P, dP)

function evaluate!(Y, tmp, SH::AbstractSHBasis, R::SVector{3})
	L = maxL(SH)
	@assert length(Y) >= sizeY(L)
	S = cart2spher(R)
	evaluate!(tmp.P, nothing, SH.alp, S)
	_evaluate!(Y, L, S, tmp.P, SH)
	return Y
end

function evaluate!(Y, SH::AbstractSHBasis{T}, R::SVector{3}) where {T} 
	L=SH.maxL
	@assert length(Y) >= sizeY(L)
	S = cart2spher(R)
	P = ACE.acquire!( P_pool, sizeP(SH.maxL) )
	compute_p!(L, S, SH.coeff, P)
	_evaluate!(Y, L, S, P, SH)
	return Y
end

function evaluate_d!(dY, tmp, SH::AbstractSHBasis, R::SVector{3})
	L = SH.maxL
	S = cart2spher(R)
	compute_dp!(L, S, SH.coeff, tmp.P, tmp.dP)
	_evaluate_d!(dY, L, S, tmp.P, tmp.dP, SH)
end

function evaluate_d!(dY, SH::AbstractSHBasis, R::SVector{3})
	L = SH.maxL
	S = cart2spher(R)
	P = ACE.acquire!( P_pool, sizeP(SH.maxL) )
	dP = ACE.acquire!( dP_pool, sizeP(SH.maxL) )
	compute_dp!(L, S, SH.coeff, P, dP)
	_evaluate_d!(dY, L, S, P, dP, SH)
end


function evaluate_ed!(Y, dY, tmp, SH::AbstractSHBasis, R::SVector{3})
	L=SH.maxL
	@assert 0 <= L <= SH.maxL
	@assert length(Y) >= sizeY(L)
	# if R[1]^2+R[2]^2 < 1e-20 * R[3]^2
	# 	R = SVector(R[1]+1e-9, R[2], R[3])
	# end
	S = cart2spher(R)
	compute_dp!(L, S, SH.coeff, tmp.P, tmp.dP)
	_evaluate_ed!(Y, dY, L, S, tmp.P, tmp.dP, SH)
	# return Y, dY
end



end

