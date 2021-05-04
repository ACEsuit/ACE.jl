
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


# k : r-degree
# l : θ-degree
# m : z-degree

import ACE: alloc_B, alloc_dB
import Base: ==
import JuLIP: evaluate!, evaluate_d!,
              alloc_temp, alloc_temp_d,
              read_dict, write_dict

import JuLIP.MLIPs: IPBasis

"""
`struct EnvPairBasis`

The basis type for environment-dependent pair potentials. This models functions
of the form
```
   V(R; {Rⱼ})
```
where `R` is the bond-direction and `{Rⱼ}` the environment. V has the following
symmetries:
 - permutations of the environment
 - rotation of the environment about the R axis
 - rotation or reflection of the entire configuration

### Construction

Use `ACE.Bonds.envpairbasis` to construct a basis.

### Evaluation

To evaluate a basis with bond-direction `R` and environment
`Renv = [R1, R2,...]` use
```
evaluate(basis, R, Renv)            # allocating
evaluate!(B, tmp, basis, R, Renv)   # non-allocating
```

### After fitting parameters, to evaluate the resulting potential

<!!!TODO!!!>
"""
struct EnvPairBasis{T0, TR, TZ, TT, TI} <: IPBasis
   P0::T0                # basis for the bond-length coordinate / m0 = k0
   Pr::TR                # specifies the radial basis  / n = kr
   Pθ::TT                # the angular basis           / l = kθ
   Pz::TZ                # specifies the z-basis       / m = kz
   aalist::BondAAList{TI}    # datastructure specifying the basis
end

Base.length(basis::EnvPairBasis) = length(basis.aalist)

fltype(basis::EnvPairBasis) = Complex{Float64}   # fltype(basis.Pr)

alloc_B(basis::EnvPairBasis, args...) = zeros(fltype(basis), length(basis))

alloc_temp(basis::EnvPairBasis, args...) =
   ( P0 = alloc_B(basis.P0),
     tmp_P0 = alloc_temp(basis.P0),
     Pr = alloc_B(basis.Pr),
     tmp_Pr = alloc_temp(basis.Pr),
     Pθ = alloc_B(basis.Pθ),
     tmp_Pθ = alloc_temp(basis.Pθ),
     Pz = alloc_B(basis.Pz),
     tmp_Pz = alloc_temp(basis.Pz),
     A = alloc_A(basis.aalist.alist)
    )

function precompute_A!(A, tmp, basis::EnvPairBasis, R0, Rs)
   alist = basis.aalist.alist
   fill!(A, 0)
   # construct the coordinate system, and convert
   cylcoords = CylindricalCoordinateSystem(R0, R0/2)
   # loop through the environment to assemble the As
   for R in Rs
      rθz = cylcoords(R)
      evaluate!(tmp.Pr, tmp.tmp_Pr, basis.Pr, rθz.r)
      evaluate!(tmp.Pz, tmp.tmp_Pz, basis.Pz, rθz.z)
      evaluate!(tmp.Pθ, tmp.tmp_Pθ, basis.Pθ, rθz)
      for i = 1:length(alist)
         krθz = alist[i]
         A[i] += tmp.Pr[krθz.kr+1] *
                 tmp.Pθ[cyl_l2i(krθz.kθ, basis.Pθ)] *
                 tmp.Pz[krθz.kz+1]
      end
      # iz = z2i(ship, Z)
      # for i = alist.firstz[iz]:(alist.firstz[iz+1]-1)
      #    zklm = alist[i]
      #    A[i] += tmp.Pr[zklm.k+1] * tmp.Pθ[cyl_l2i(zklm.l)] * tmp.Pz[zklm.m+1]
      # end
   end
   return A
end

# R0   : typically SVector{T}
# Renv : typically Vector{SVector{T}} or a view into Vector{SVector{T}}
function evaluate!(B::AbstractVector{Complex{T}},
                   tmp, basis::EnvPairBasis, R0, Renv)  where {T}
   aalist = basis.aalist
   # construct the basis for the r1-variable
   r0 = norm(R0)
   P0 = evaluate!(tmp.P0, tmp.tmp_P0, basis.P0, r0)

   # evaluate the A-basis, i.e. the density projections of the environment
   A = precompute_A!(tmp.A, tmp, basis, R0, Renv)

   # loop over all basis functions
   for i = 1:length(aalist)
      B[i] = P0[aalist.i2k0[i] + 1]
      for α = 1:aalist.len[i]
         B[i] *= A[aalist.i2Aidx[i, α]]  # TODO: reverse i, α
      end
   end

   return B
end


# ------------------------------------------------------------


using JuLIP: @pot, AbstractCalculator
import JuLIP.MLIPs: combine

"""
A rudimentary implementation of an environment-dependent pair potential.
"""
struct EnvPairPot{T, T0, TR, TZ, TT, TI} <: AbstractCalculator
   basis::EnvPairBasis{T0, TR, TZ, TT, TI}
   c::Vector{T}
end

@pot EnvPairPot

combine(b::EnvPairBasis, c::AbstractVector{<: Number}) = EnvPairPot(b, c)

alloc_temp(V::EnvPairPot, args...) = alloc_temp(V.basis, args...)

# R0   : typically SVector{T}
# Renv : typically Vector{SVector{T}} or a view into Vector{SVector{T}}
function evaluate!(tmp, V::EnvPairPot, R0, Renv)
   val = zero(ComplexF64)

   aalist = V.basis.aalist
   # construct the basis for the r1-variable
   r0 = norm(R0)
   P0 = evaluate!(tmp.P0, tmp.tmp_P0, V.basis.P0, r0)
   # evaluate the A-basis, i.e. the density projections of the environment
   A = precompute_A!(tmp.A, tmp, V.basis, R0, Renv)

   # loop over all basis functions
   for i = 1:length(aalist)
      b = P0[aalist.i2k0[i] + 1]
      for α = 1:aalist.len[i]
         b *= A[aalist.i2Aidx[i, α]]
      end
      val += V.c[i] * b
   end

   return real(val)
end


# --------------(de-)serialisation----------------------------------------

import JuLIP: write_dict, read_dict

write_dict(b::EnvPairBasis) =
      Dict( "__id__" => "ACE_EnvPairBasis",
            "P0"     => write_dict(b.P0),
            "Pr"     => write_dict(b.Pr),
            "Ptheta" => write_dict(b.Pθ),
            "Pz"     => write_dict(b.Pz),
            "aalist" => write_dict(b.aalist) )

read_dict(::Val{:ACE_EnvPairBasis}, D::Dict) =
      EnvPairBasis(read_dict(D["P0"]),
                   read_dict(D["Pr"]),
                   read_dict(D["Ptheta"]),
                   read_dict(D["Pz"]),
                   read_dict(D["aalist"]))

write_dict(V::EnvPairPot) =
      Dict( "__id__" => "ACE_EnvPairPot",
            "basis" => write_dict(V.basis),
            "cr" => real.(V.c),
            "ci" => imag.(V.c) )

read_dict(::Val{:ACE_EnvPairPot}, D::Dict) =
      EnvPairPot( read_dict(D["basis"]),
                  D["cr"] + im * D["ci"] )

==(B1::EnvPairBasis, B2::EnvPairBasis) =
         all(getfield(B1, x) == getfield(B2, x) for x in fieldnames(EnvPairBasis))

==(B1::EnvPairPot, B2::EnvPairPot) =
         all(getfield(B1, x) == getfield(B2, x) for x in fieldnames(EnvPairPot))
