
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------




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

Base.length(basis::BasicPSH1pBasis, z0::AtomicNumber) =
      numz(basis) *  length(basis.spec)

Base.length(basis::BasicPSH1pBasis, iz0::Integer) =
      numz(basis) *  length(basis.spec)

Base.length(basis::BasicPSH1pBasis, iz::Integer, iz0::Integer) =
      length(basis.spec)



function get_basis_spec(basis::BasicPSH1pBasis, z0::AtomicNumber)
   iz0 = z2i(basis, z0)
   len_iz0 = sum(length(basis, iz, iz0) for iz = 1:numz(basis))
   spec = Vector{PSH1pBasisFcn}(undef, len_iz0)
   for iz = 1:numz(basis)
      z = i2z(basis, iz)
      spec[basis.Aindices[iz, iz0]] =
               [ PSH1pBasisFcn(b.n, b.l, b.m, z)   for b in basis.spec ]
   end
   return spec
end

get_basis_spec(basis::BasicPSH1pBasis, iz0::Integer, i::Integer) =
   get_basis_spec(basis, i2z(basis, iz0), i)

function get_basis_spec(basis::BasicPSH1pBasis, z0::AtomicNumber, i::Integer)
   iz = ((i-1) ÷ length(basis.spec)) + 1
   inew = mod1(i, length(basis.spec))
   b = basis.spec[inew]
   return PSH1pBasisFcn(b.n, b.l, b.m, i2z(basis, iz))
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


@doc raw"""
`function _get_PSH_1p_spec`

Construct the specification for a ``P \otimes Y`` type 1-particle basis.
These must be treated differently because of the requirements that complete
``l``-blocks are represented in the basis.

See also: `_get_1p_spec`.
"""
function _get_PSH_1p_spec(J::ScalarBasis, D::AbstractDegree)
   # find out what the largest degree is that we can allow:
   maxdeg = maximum(D(PSH1pBasisFcn(n, 0, 0, 0)) for n = 1:length(J))

   # generate the `spec::Vector{PSH1pBasisFcn}` using length(J)
   specnl = gensparse(2, maxdeg;
                      tup2b = t -> PSH1pBasisFcn(t[1]+1, t[2], 0, 0),
                      degfun = t -> D(t),
                      ordered = false)
   # add the m-parameters
   return [ PSH1pBasisFcn(b.n, b.l, m, 0)
              for b in specnl for m = -b.l:b.l ]
end




# function precompute_dA!(A, dA, tmp, alist, Rs, Zs, ship)
#    fill!(A, 0)
#    fill!(dA, zero(eltype(dA)))
#    for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
#       # precompute the derivatives of the Jacobi polynomials and Ylms
#       evaluate_d!(tmp.J, tmp.dJ, tmp.tmpJ, ship.J, norm(R))
#       evaluate_d!(tmp.Y, tmp.dY, tmp.tmpY, ship.SH, R)
#       # deduce the A and dA values
#       iz = z2i(ship, Z)
#       R̂ = R / norm(R)
#       for i = alist.firstz[iz]:(alist.firstz[iz+1]-1)
#          zklm = alist[i]
#          ik = zklm.k+1; iy = index_y(zklm.l, zklm.m)
#          A[i] += tmp.J[ik] * tmp.Y[iy]
#          # and into dA # grad_phi_Rj(R, iR, zklm, tmp)
#          ∇ϕ_zklm = tmp.dJ[ik] * tmp.Y[iy] * R̂ + tmp.J[ik] * tmp.dY[iy]
#          dA[iR, i] = ∇ϕ_zklm
#       end
#    end
#    return dA
# end
