
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------




# ----------------------------------------------------------------------
# The repulsive core is built from shifted Buckingham potentials

@doc raw"""
`struct BuckPot` : Buckingham potential,
```math
   V(r) = e_0 + B  \exp\big( - A (r/r_i - 1)\big) \cdot \frac{r_i}{r}
```
"""
struct BuckPot{T} <: SimplePairPotential
   e0::T
   A::T
   ri::T
   B::T
end

@pot BuckPot

evaluate(V::BuckPot, r::Number) = V.e0 + V.B * exp( - V.A * (r/V.ri-1) ) * V.ri/r

evaluate_d(V::BuckPot, r::Number) = V.B * exp( - V.A * (r/V.ri-1) ) * V.ri * (
                            - V.A / V.ri / r  - 1/r^2  )

write_dict(V::BuckPot) = Dict(
         "__id__" => "ACE_BuckPot",
             "e0" => V.e0,
              "A" => V.A,
             "ri" => V.ri,
              "B" => V.B   )

read_dict(::Val{:SHIPs_BuckPot}, D::Dict) =
   read_dict(Val{:ACE_BuckPot}(), D)

read_dict(::Val{:ACE_BuckPot}, D::Dict) =
      BuckPot(D["e0"], D["A"], D["ri"], D["B"])

==(V1::BuckPot, V2::BuckPot) = _allfieldsequal(V1, V2)


# ----------------------------------------------------------------------


struct RepulsiveCore{T, TOUT, NZ} <: PairPotential
   Vout::TOUT
   Vin::SMatrix{NZ, NZ, BuckPot{T}}
end

@pot RepulsiveCore

==(V1::RepulsiveCore, V2::RepulsiveCore) = _allfieldsequal(V1, V2)

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
   if dv >= 0.0 && verbose
      @warn("The slope `Vout'(ri)` should be negative")
   end
   if dv > -1.0 && verbose
      @warn("""The slope `Vout'(ri) = $dv` may not be steep enough to attach a
               repulsive core. Proceed at your own risk.""")
   end
   if v-e0 <= 0.0 && verbose
      @warn("it is recommended that `Vout(ri) > 0`.")
   end
   if v-e0 <= 1.0 && verbose
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
   nz = numz(Vout)
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
   nz = numz(Vout)
   return RepulsiveCore(Vout, SMatrix{nz, nz}(Vin...))
end


function RepulsiveCore(Vout, D::Dict; verbose=false)
   _sort_key(V, key) = tuple(sort([z2i.(Ref(V), AtomicNumber.(key))...])...)
   Di = Dict([_sort_key(Vout, key) => val for (key, val) in D]...)
   nz = numz(Vout)
   Vin = Matrix{Any}(undef, nz, nz)
   for i0 = 1:nz, i1 = 1:i0
      z0, z1 = i2z(Vout, i0), i2z(Vout, i1)
      ri, e0 = Di[(i1, i0)].ri, Di[(i1, i0)].e0
      Vin[i0, i1] = Vin[i1, i0] =
            _simple_repulsive_core(Vout, ri, e0, verbose, z1, z0)
   end
   # construct the piecewise potential
   return RepulsiveCore(Vout, Vin)
end

# ----------------------------------------------------
#  File IO

write_dict(V::RepulsiveCore) = Dict(
      "__id__" => "ACE_RepulsiveCore",
        "Vout" => write_dict(V.Vout),
         "Vin" => write_dict.(V.Vin[:]) )

function read_dict(::Val{:ACE_RepulsiveCore}, D::Dict)
   Vout = read_dict(D["Vout"])
   nz = numz(Vout)
   Vin = read_dict.(D["Vin"])
   return RepulsiveCore(Vout, reshape(Vin, (nz, nz)))
end
