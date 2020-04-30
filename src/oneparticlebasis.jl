
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
      end
   end
   return basis
end


# kz2iA(A::AwithViews, k, iz::Integer, iz0::Integer) = A.indices[iz, iz0][k]



@doc raw"""
`struct BasicPSH1pBasis <: OneParticleBasis`

One-particle basis of the form
```math
\phi_{nlm}({\bm r}, z_1, z_0) = J_{n}(r) Y_l^m(\hat{\br r})
```
where ``J_{n}`` denotes a radial basis.
"""
struct BasicPSH1pBasis{T, NZ, TJ <: ScalarBasis{T}} <: OneParticleBasis{T}
   J::TJ
   SH::SHBasis{T}
   zlist::SZList{NZ}
   spec::Vector{PSH1pBasisFcn}
   Aindices::Matrix{UnitRange{Int}}
end


function BasicPSH1pBasis(J::ScalarBasis{T};
                         species = :X,
                         D::AbstractDegree = SparsePSHDegree()
                ) where {T}
   # get a generic basis spec
   spec = _get_PSH_1p_spec(J::ScalarBasis, D::AbstractDegree)
   # now get the maximum L-degree to generate the SH basis
   maxL = maximum(b.l for b in spec)
   SH = SHBasis(maxL, T)
   # construct the basis
   zlist = ZList(species; static=true)
   NZ = length(zlist)
   P = BasicPSH1pBasis(J, SH, zlist, spec,
                       Matrix{UnitRange{Int}}(undef, NZ, NZ))
   set_Aindices!(P)
   return P
end

Base.length(basis::BasicPSH1pBasis, iz::Integer, iz0::Integer) =
      length(basis.spec)


function get_basis_spec(basis::BasicPSH1pBasis, z0::AtomicNumber)
   iz0 = z2i(basis, iz0)
   len_iz0 = sum(length(basis, iz) for iz = 1:numz(basis))
   spec = Vector{PSH1pBasisFcn}(undef, len_iz0)
   for iz = 1:numz(basis)
      spec[basis.Aindices[iz, iz0]] =
               [ PSH1pBasisFcn(b.n, b.l, b.m, iz)   for b in basis.spec ]
   end
   return spec
end


Base.eltype(basis::BasicPSH1pBasis{T}) where T = Complex{T}
reltype(basis::BasicPSH1pBasis{T}) where T = T
# eltype and length should provide automatic allocation of alloc_B, alloc_dB

alloc_temp(basis::BasicPSH1pBasis, args...) =
      ( BJ = alloc_B(basis.J, args...),
        tmpJ = alloc_temp(basis.J, args...),
        BY = alloc_B(basis.SH, args...),
        tmpY = alloc_temp(basis.SH, args...),
       )

function add_into_A!(A, tmp, basis::BasicPSH1pBasis,
                     R, iz::Integer, iz0::Integer)
   # evaluate the r-basis and the R̂-basis for the current neighbour at R
   evaluate!(tmp.BJ, tmp.tmpJ, basis.J, norm(R))
   evaluate!(tmp.BY, tmp.tmpY, basis.SH, R)
   # add the contributions to the A_zklm
   for (i, nlm) in enumerate(basis.spec)
      A[i] += tmp.BJ[nlm.n] * tmp.BY[index_y(nlm.l, nlm.m)]
   end
   return nothing
end
