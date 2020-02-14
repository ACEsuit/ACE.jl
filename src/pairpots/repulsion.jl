

module Repulsion

using StaticArrays

import JuLIP: decode_dict
import JuLIP.Potentials: @pot, evaluate, evaluate_d, MPairPotential, @D, cutoff,
                         @analytic, evaluate!, evaluate_d!,
                         alloc_temp, alloc_temp_d,
                         i2z, z2i
import Base: Dict, convert, ==

# ----------------------------------------------------------------------
# The repulsive core is built from shifted Buckingham potentials

"""
e0 + B * exp( - A * (r/ri-1) ) * ri/r
"""
struct BuckPot{T}
   e0::T
   A::T
   ri::T
   B::T
end

@pot BuckPot

evaluate(V::BuckPot, r) = V.e0 + V.B * exp( - V.A * (r/V.ri-1) ) * V.ri/r

evaluate_d(V::BuckPot, r) = V.B * exp( - V.A * (r/V.ri-1) ) * V.ri * (
                            - V.A / V.ri / r  - 1/r^2  )

Dict(V::BuckPot) = Dict("__id__" => "PolyPairPots_BuckPot",
                        "e0" => V.e0, "A" => V.A, "ri" => V.ri, "B" => V.B )

BuckPot(D::Dict) = BuckPot(D["e0"], D["A"], D["ri"], D["B"])

convert(::Val{:PolyPairPots_BuckPot}, D::Dict) = BuckPot(D)

==(V1::BuckPot, V2::BuckPot) =
   all(getfield(V1, x) == getfield(V2, x) for x in fieldnames(BuckPot))

# ----------------------------------------------------------------------


struct RepulsiveCore{T, TOUT, NZ} <: MPairPotential
   Vout::TOUT
   Vin::SMatrix{NZ, NZ, BuckPot{T}}
end

==(V1::RepulsiveCore, V2::RepulsiveCore) =
   (V1.Vout == V2.Vout) && (V1.Vin == V2.Vin)


@pot RepulsiveCore

cutoff(V::RepulsiveCore) = cutoff(V.Vout)

alloc_temp(V::RepulsiveCore, N::Integer) = alloc_temp(V.Vout, N)
alloc_temp_d(V::RepulsiveCore, N::Integer) = alloc_temp_d(V.Vout, N)

function evaluate!(tmp, V::RepulsiveCore, r::Number, z, z0)
   iz, iz0 = z2i(V.Vout, z), z2i(V.Vout, z0)
   Vin = V.Vin[iz, iz0]
   if r > Vin.ri
      return evaluate!(tmp, V.Vout, r, z, z0)
   else
      return evaluate(Vin, r)
   end
end

function evaluate_d!(tmp, V::RepulsiveCore, r::Number, z, z0)
   iz, iz0 = z2i(V.Vout, z), z2i(V.Vout, z0)
   Vin = V.Vin[iz, iz0]
   if r > Vin.ri
      return evaluate_d!(tmp, V.Vout, r, z, z0)
   else
      return evaluate_d(Vin, r)
   end
end


function _simple_repulsive_core(Vout, ri, e0, verbose, z, z0)
   v = Vout(ri, z, z0)
   dv = @D Vout(ri, z, z0)
   if dv >= 0.0
      @warn("The slope `Vout'(ri)` should be negative")
   end
   if dv > -1.0
      @warn("""The slope `Vout'(ri) = $dv` may not be steep enough to attach a
               repulsive core. Proceed at your own risk.""")
   end
   if v-e0 <= 0.0
      @warn("it is recommended that `Vout(ri) > 0`.")
   end
   if v-e0 <= 1.0
      @warn("""Ideally the repulsive core should not be attached at small
               values of `Vout(ri) = $v`. Proceed at your own risk.""")
   end
   # e0 + B e^{-A (r/ri-1)} * ri/r
   #    => e0 + B = Vout(ri) => = Vout(ri) - e0 = v - e0
   # dv = - A*B/ri e^{-A (r/ri-1)} * ri/r - B*ri*e^{...} / r^2
   #    = - A/ri * (v - 1/ri * (v = - (1+A)/ri * (v-e0)
   #    => -(1+A)/ri * (v-e0) = dv
   #    => 1+A = - ri dv / (v-e0)
   #    => A = -1 - ri dv / (v-e0)
   A = -1 - ri * dv / (v-e0)
   B = v-e0
   Vin = BuckPot(e0, A, ri, B)
   if verbose
      @show ri
      @show Vout(ri), (@D Vout(ri))
      @show Vin(ri), (@D Vin(ri))
   end
   return Vin
end

function RepulsiveCore(Vout, ri::Number, e0=0.0; verbose=false)
   nz = length(Vout.zlist)
   Vin = Matrix{Any}(undef, nz, nz)
   for i0 = 1:nz, i1 = 1:i0
      z0, z1 = i2z(Vout, i0), i2z(Vout, i1)
      Vin[i0, i1] = Vin[i1, i0] =
            _simple_repulsive_core(Vout, ri, e0, verbose, z1, z0)
   end
   # construct the piecewise potential
   return RepulsiveCore(Vout, Vin)
end

function RepulsiveCore(Vout, Vin::AbstractArray)
   nz = length(Vout.zlist)
   return RepulsiveCore(Vout, SMatrix{nz, nz}(Vin...))
end

# ----------------------------------------------------
#  File IO
# ----------------------------------------------------

Dict(V::RepulsiveCore) = Dict("__id__" => "PolyPairPots_RepulsiveCore",
                              "Vout" => Dict(V.Vout),
                              "Vin" => Dict.(V.Vin[:]) )

function RepulsiveCore(D::Dict)
   Vout = decode_dict(D["Vout"])
   nz = length(Vout.zlist)
   Vin = BuckPot.(D["Vin"])
   return RepulsiveCore(Vout, reshape(Vin, (nz, nz)))
end

convert(::Val{:PolyPairPots_RepulsiveCore}, D::Dict) = RepulsiveCore(D)

end
