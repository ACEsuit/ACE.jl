
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using StaticArrays

using SHIPs.SphericalHarmonics: SHBasis, index_y
using JuLIP.Potentials: ZList, SZList, numz


function evaluate!(A, tmp, basis::OneParticleBasis, Rs, Zs::AbstractVector, z0)
   fill!(A, 0)
   iz0 = z2i(basis, z0)
   for (R, Z) in zip(Rs, Zs)
      iz = z2i(basis, Z)
      add_into_A!((@view A[basis.Aindices[iz, iz0]]), tmp, basis, R, iz, iz0)
   end
   return A
end

function evaluate!(A, tmp, basis::OneParticleBasis, R, z::AtomicNumber, z0)
   fill!(A, 0)
   iz0, iz = z2i(basis, z0), z2i(basis, z)
   add_into_A!((@view A[basis.Aindices[iz, iz0]]), tmp, basis, R, iz, iz0)
   return A
end


function alloc_B(basis::OneParticleBasis, args...)
   NZ = numz(basis)
   maxlen = maximum( sum( length(basis, iz, iz0) for iz = 1:NZ )
                     for iz0 = 1:NZ )
   T = eltype(basis)
   return zeros(T, maxlen)
end


function set_Aindices!(basis::OneParticleBasis)
   NZ = numz(basis)
   for iz0 = 1:NZ
      idx = 0
      for iz = 1:NZ
         len = length(basis, iz, iz0)
         basis.Aindices[iz, iz0] = (idx+1):(idx+len)
         idx += len
      end
   end
   return basis
end

get_basis_spec(basis::OneParticleBasis, iz0::Integer) =
      get_basis_spec(basis, i2z(iz0))

get_basis_spec(basis::OneParticleBasis, s::Symbol) =
      get_basis_spec(basis, atomic_number(s))
