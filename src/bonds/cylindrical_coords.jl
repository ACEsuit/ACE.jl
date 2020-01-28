
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using LinearAlgebra: cross, norm, dot
using StaticArrays


struct CylindricalCoordinateSystem{T}
   ez::SVector{3, T}
   ex::SVector{3, T}
   ey::SVector{3, T}
end

struct CylindricalCoordinates{T}
   cosθ::T
   sinθ::T
   r::T
   z::T
end


function CylindricalCoordinateSystem(R::AbstractVector{T}) where {T}
   @assert length(R) == 3
   # first coordinate
   ez = R / norm(R)
   # second coordinate
   if abs(ez[1] - 1) > 0.3
      ex = @SVector T[1, 0, 0]
   else
      ex = @SVector T[0, 1, 0]
   end
   ex = ex - dot(ez, ex) * ez
   ex /= norm(ex)
   # third coordinate
   ey = cross(ex, ez)
   return CylindricalCoordinateSystem(ez, ex, ey)
end


(C::CylindricalCoordinateSystem)(R) = cylindrical(C, R)

function cylindrical(C::CylindricalCoordinateSystem{T}, R::AbstractVector{T}) where {T}
   @assert length(R) == 3
   x = dot(R, C.ex)
   y = dot(R, C.ey)
   z = dot(R, C.ez)
   r = sqrt(x^2 + y^2)
   cosθ = x / r
   sinθ = y / r
   return CylindricalCoordinates(cosθ, sinθ, r, z)
end

cartesian(C::CylindricalCoordinateSystem{T},
          c::CylindricalCoordinates{T}) where {T} =
      c.r * c.cosθ * C.ex + c.r * c.sinθ * C.ey + c.z * C.ez


# ------------------------------------------------------------
#   Fourier Basis evaluation

struct FourierBasis{T}
   deg::Int
   _fltt::Type{T}
   FourierBasis(deg::Integer,
                fltt::Type{<: AbstractFloat} = Float64) = new(deg, fltt)
end

Base.eltype(::FourierBasis{T}) where {T} = T
Base.length(fB::FourierBasis) = 2 * fB.deg + 1

Dict(fB::FourierBasis) = Dict(
      "__id__" => "PoSH_FourierBasis",
      "deg" => deg,
      "fltt" => "$(fG._fltt)"
   )

FourierBasis(D::Dict) = FourierBasis(D["deg"], D["fltt"])
FourierBasis(deg::Integer, fltt::AbstractString) =
      FourierBasis(deg, eval(Meta.parse(fltt)))

convert(::Val{:PoSH_FourierBasis}, D::Dict) = FourierBasis(D)


alloc_B(fB::FourierBasis) = zeros(length(fB))
alloc_dB(fB::FourierBasis, args...) = zeros(JVec{T}, length(fB))

# specify ordering
cyl_l2i(l, maxL) = maxL + 1 + l  # = i
cyl_i2l(i, maxL) = i - maxL - 1

function evaluate!(P, fB::FourierBasis, c::CylindricalCoordinates{T}
                  ) where {T}
   @assert length(P) >= length(fB)
   z = c.cosθ + im * c.sinθ
   zl = one(T) + im * zero(T)
   P[_l2i(0)] = zl
   for l = 1:maxL
      zl *= z
      P[_l2i( l)] = zl
      P[_l2i(-l)] = conj(zl)
   end
   return P
end

# we only return ∂Pl/∂x̂ since
#     ∂Pl/∂ŷ = ∂Pl/∂z * ∂z / ∂ŷ
#            = im * ∂Pl/∂z
#            = im * ∂Pl/∂x̂
function evaluate_d!(P, dP, fB::FourierBasis, c::CylindricalCoordinates{T}
                  ) where {T}
   @assert length(P) >= length(fB)
   z = c.cosθ + im * c.sinθ    # z = x̂ + i ŷ
   zl = one(T) + im * zero(T)  # zl = z^l -> initialise to z^0 = 1
   P[_l2i(0)] = zl
   dP[_l2i(0)] =  zero(T)
   for l = 1:maxL
      # ∂P_{ l}/∂x̂ = l z^(l-1)
      # ∂P_{-l}/∂x̂ = l z̄^(l-1)
      dP[_l2i( l)] = l * zl
      dP[_l2i(-l)] = l * conj(zl)
      zl *= z  # zl = z^l
      P[_l2i( l)] = zl
      P[_l2i(-l)] = conj(zl)
      dP
   end
   return P
end
