
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using LinearAlgebra: cross, norm, dot
using StaticArrays


struct CylindricalCoordinateSystem{T}
   ez::SVector{3, T}
   ex::SVector{3, T}
   ey::SVector{3, T}
   o::SVector{3, T}
end

struct CylindricalCoordinates{T}
   cosθ::T
   sinθ::T
   r::T
   z::T
end


function CylindricalCoordinateSystem(R::AbstractVector{T},
                                     o=zero(SVector{3,T})) where {T}
   @assert length(R) == 3
   # first coordinate
   ez = R / norm(R)
   # second coordinate
   if abs(ez[1] - 1) > 0.33
      ex = @SVector T[1, 0, 0]
   else
      ex = @SVector T[0, 1, 0]
   end
   ex = ex - dot(ez, ex) * ez
   ex /= norm(ex)
   # third coordinate
   ey = cross(ex, ez)
   return CylindricalCoordinateSystem(ez, ex, ey, o)
end


(C::CylindricalCoordinateSystem)(R) = cylindrical(C, R)

function cylindrical(C::CylindricalCoordinateSystem{T},
                     R::AbstractVector{T}) where {T}
   @assert length(R) == 3
   Rc = R - C.o
   x = dot(Rc, C.ex)
   y = dot(Rc, C.ey)
   z = dot(Rc, C.ez)
   r = sqrt(x^2 + y^2)
   cosθ = x / r
   sinθ = y / r
   return CylindricalCoordinates(cosθ, sinθ, r, z)
end

cartesian(C::CylindricalCoordinateSystem{T},
          c::CylindricalCoordinates{T}) where {T} =
      C.o + (c.r * c.cosθ * C.ex + c.r * c.sinθ * C.ey + c.z * C.ez)
