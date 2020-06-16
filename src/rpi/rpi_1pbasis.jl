
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------






@doc raw"""
`struct PSH1pBasis <: OneParticleBasis`

One-particle basis of the form
```math
\begin{aligned}
\phi_{nlm}({\bm r}, z_1, z_0) &= P_{nl}(r) Y_l^m(\hat{\br r}) \\ 
P_{nl}(r) &= C_{nl,n'} J_{n'}(r)
\end{aligned}
```
where ``J_{n}`` denotes a radial basis.
"""
struct PSH1pBasis{T, NZ, TJ <: ScalarBasis{T}} <: OneParticleBasis{T}
   J::TJ
   SH::SHBasis{T}
   zlist::SZList{NZ}
   C::SMatrix{NZ, NZ, Matrix{T}}
   spec::SMatrix{NZ, NZ, Vector{PSH1pBasisFcn}}
   Aindices::SMatrix{NZ, NZ, UnitRange{Int}}
   nlmap::SMatrix{NZ, NZ, Matrix{Int}}
end


# --------------------------------------------------
# lots of basic book-keeping 

"largest (P^{z z0}_nl)_{nl} sub-basis"
_maxnl(basis::PSH1pBasis) = maximum(size(C, 1) for C in basis.C)

"index into BY (spherical harmonics)"
_idxY(b1p::PSH1pBasis, nlm, iz, iz0) = index_y(nlm.l, nlm.m)

"index into BP (Pnl radial basis)"
_idxR(b1p::PSH1pBasis, nlm, iz, iz0) = idxR(b1p.nlmap[iz, iz0][nlm.n, nlm.l])

get_basis_spec(basis::PSH1pBasis, z0::AtomicNumber, args...) = 
      get_basis_spec(basis, z2i(basis, z0), args...)

get_basis_spec(basis::PSH1pBasis, iz0::Integer) = 
      vcat( basis.spec[:, iz0]... )

function get_basis_spec(basis::PSH1pBasis, iz0::Integer, i::Integer)
   idx = 0
   for iz = 1:numz(basis) 
      if idx + length(basis.spec[iz, iz0]) >= i 
         return basis.spec[iz, iz0][i - idx]
      end 
   end 
   error("index i = $i not found; probably this means that i > length(basis[iz0])")
end

cutoff(basis::PSH1pBasis) = cutoff(basis.J)


Base.length(basis::PSH1pBasis, z::AtomicNumber, z0::AtomicNumber) =
      length(basis, z2i(basis, z). z2i(basis, z0))

Base.length(basis::BasicPSH1pBasis, iz::Integer, iz0::Integer) =
      length(basis.spec[iz, iz0])

Base.length(basis::PSH1pBasis, z0::AtomicNumber) =
      length(basis, z2i(basis, z0))

Base.length(basis::PSH1pBasis, iz0::Integer) =
      sum(length(basis, iz, iz0) for iz = 1:numz(basis))


# ----------------------------------------
# basis construction 

# function PSH1pBasis(species, )
# end 


# function BasicPSH1pBasis(J::ScalarBasis{T}, zlist::SZList,
#                          spec::Vector{PSH1pBasisFcn}) where {T}
#    # now get the maximum L-degree to generate the SH basis
#    maxL = maximum(b.l for b in spec)
#    SH = SHBasis(maxL, T)
#    # construct a preliminary Aindices array to get an incorrect basis
#    NZ = length(zlist)
#    P = BasicPSH1pBasis(J, SH, zlist, spec,
#                        Matrix{UnitRange{Int}}(undef, NZ, NZ))
#    # ... now fix the Aindices array
#    set_Aindices!(P)
#    return P
# end

# function BasicPSH1pBasis(J::ScalarBasis;
#                          species = :X,
#                          D::AbstractDegree = SparsePSHDegree())
#    # get a generic basis spec
#    spec = _get_PSH_1p_spec(J::ScalarBasis, D::AbstractDegree, species)
#    # construct the basis
#    zlist = ZList(species; static=true)
#    return BasicPSH1pBasis(J, zlist, spec)
# end






# # ------------------------------------------------------
# #  FIO code


# ==(P1::BasicPSH1pBasis, P2::BasicPSH1pBasis) =  SHIPs._allfieldsequal(P1, P2)

# write_dict(basis::BasicPSH1pBasis{T}) where {T} = Dict(
#       "__id__" => "SHIPs_BasicPSH1pBasis",
#            "J" => write_dict(basis.J),
#           "SH" => write_dict(basis.SH),
#         "spec" => write_dict.(basis.spec),
#        "zlist" => write_dict(basis.zlist),
#    )

# function read_dict(::Val{:SHIPs_BasicPSH1pBasis}, D::Dict)
#    J = read_dict(D["J"])
#    SH = read_dict(D["SH"])
#    zlist = read_dict(D["zlist"])
#    spec = read_dict.(D["spec"])
#    P = BasicPSH1pBasis(J, SH, zlist, spec,
#                  Matrix{UnitRange{Int}}(undef, length(zlist), length(zlist)))
#    set_Aindices!(P)
#    return P
# end



# ------------------------------------------------------
#  Evaluation code


Base.eltype(basis::PSH1pBasis{T}) where T = Complex{T}
reltype(basis::BasicPSH1pBasis{T}) where T = T
# eltype and length should provide automatic allocation of alloc_B, alloc_dB


alloc_temp(basis::PSH1pBasis, args...) =
      (
        BJ = alloc_B(basis.J, args...),
        BP = zeros(eltype(basis), _maxnl(basis)), 
        BY = alloc_B(basis.SH, args...),
        tmpJ = alloc_temp(basis.J, args...),
        tmpY = alloc_temp(basis.SH, args...),
       )



function add_into_A!(A, tmp, basis::PSH1pBasis,
                     R, iz::Integer, iz0::Integer)
   # evaluate the r-basis 
   C = basis.C[iz, iz0]  # n' -> nl transformation coefficients
   maxnJ = size(C, 2)    # number of J basis functions (pre-basis)
   evaluate!(tmp.BJ, tmp.tmpJ, basis.J, norm(R); maxn=maxnJ)
   maxn = size(C,1)      # number of transformed basis functions 
   BP = mul!((@view tmp.BP[1:maxn]), C, (@view tmp.BJ[1:maxnJ]))

   # evaluate the R̂-basis for the current neighbour at R
   evaluate!(tmp.BY, tmp.tmpY, basis.SH, R)

   # add the contributions to the A_zklm
   @inbounds for (i, nlm) in enumerate(basis.spec[iz, iz0])
      A[i] += (tmp.BP[_idxR(basis, nlm, iz, iz0)] 
                * tmp.BY[_idxY(basis, nlm, iz, iz0)] )
   end
   return nothing
end


# alloc_temp_d(basis::BasicPSH1pBasis, args...) =
#       (
#         BJ = alloc_B(basis.J, args...),
#         tmpJ = alloc_temp(basis.J, args...),
#         BY = alloc_B(basis.SH, args...),
#         tmpY = alloc_temp(basis.SH, args...),
#         dBJ = alloc_dB(basis.J, args...),
#         tmpdJ = alloc_temp_d(basis.J, args...),
#         dBY = alloc_dB(basis.SH, args...),
#         tmpdY = alloc_temp_d(basis.SH, args...),
#        )

# function add_into_A_dA!(A, dA, tmpd, basis::BasicPSH1pBasis, R, iz::Integer, iz0::Integer)
#    r = norm(R)
#    R̂ = R / r
#    # evaluate the r-basis and the R̂-basis for the current neighbour at R
#    evaluate_d!(tmpd.BJ, tmpd.dBJ, tmpd.tmpdJ, basis.J, r)
#    evaluate_d!(tmpd.BY, tmpd.dBY, tmpd.tmpdY, basis.SH, R)
#    # add the contributions to the A_zklm, ∇A
#    @inbounds for (i, nlm) in enumerate(basis.spec)
#       iY = index_y(nlm.l, nlm.m)
#       A[i] += tmpd.BJ[nlm.n] * tmpd.BY[iY]
#       dA[i] = (tmpd.dBJ[nlm.n] * tmpd.BY[iY]) * R̂ + tmpd.BJ[nlm.n] * tmpd.dBY[iY]
#    end
#    return nothing
# end
