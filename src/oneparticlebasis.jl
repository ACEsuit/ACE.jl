
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



function evaluate!(A, tmp, basis::OneParticleBasis, Rs, Zs::AbstractVector, z0)
   fill!(A, 0)
   iz0 = z2i(basis, z0)
   for (R, Z) in zip(Rs, Zs)
      iz = z2i(basis, Z)
      add_into_A!((@view A[basis.Aindices[iz, iz0]]), tmp, basis, R, iz, iz0)
   end
   return A
end


function evaluate_d!(A, dA, tmpd, basis::OneParticleBasis,
                     Rs, Zs::AbstractVector, z0)
   fill!(A, 0)
   fill!(dA, zero(eltype(dA)))  # TODO: this should not be necessary!
   iz0 = z2i(basis, z0)
   for (j, (R, Z)) in enumerate(zip(Rs, Zs))
      iz = z2i(basis, Z)
      Aview = @view A[basis.Aindices[iz, iz0]]
      dAview = @view dA[basis.Aindices[iz, iz0], j]
      add_into_A_dA!(Aview, dAview, tmpd, basis, R, iz, iz0)
   end
   return dA
end


function evaluate!(A, tmp, basis::OneParticleBasis, R, z::AtomicNumber, z0)
   fill!(A, 0)
   iz0, iz = z2i(basis, z0), z2i(basis, z)
   add_into_A!((@view A[basis.Aindices[iz, iz0]]), tmp, basis, R, iz, iz0)
   return A
end

function evaluate_d!(A, dA, tmpd, basis::OneParticleBasis,
                     R, z::AtomicNumber, z0)
   fill!(A, 0)
   iz, iz0 = z2i(basis, z), z2i(basis, z0)
   Aview = @view A[basis.Aindices[iz, iz0]]
   dAview = @view dA[basis.Aindices[iz, iz0]]
   add_into_A_dA!(Aview, dAview, tmpd, basis, R, iz, iz0)
   return dA
end




function alloc_B(basis::OneParticleBasis, args...)
   NZ = numz(basis)
   maxlen = maximum( sum( length(basis, iz, iz0) for iz = 1:NZ )
                     for iz0 = 1:NZ )
   T = fltype(basis)
   return zeros(T, maxlen)
end

# TODO: fix the dispatch structure on alloc_dB
alloc_dB(basis::OneParticleBasis, ::JVec) = alloc_dB(basis, 1)
alloc_dB(basis::OneParticleBasis, Rs::AbstractVector{<: JVec}, args...) =
      alloc_dB(basis, length(Rs))
alloc_dB(basis::OneParticleBasis) = alloc_dB(basis, 1)

function alloc_dB(basis::OneParticleBasis, maxN::Integer)
   NZ = numz(basis)
   maxlen = maximum( sum( length(basis, iz, iz0) for iz = 1:NZ )
                     for iz0 = 1:NZ )
   T = fltype(basis)
   return zeros(JVec{T}, (maxlen, maxN))
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
