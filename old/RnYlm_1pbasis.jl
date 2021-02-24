
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

import ACE.SphericalHarmonics: SHBasis, index_y


@doc raw"""
`struct RnYlmBasisFcn` : 1-particle basis function specification
for bases of the type ``P \otimes Y`` with `P::ScalarBasis` and `Y::SHBasis`
"""
struct RnYlmBasisFcn <: OnepBasisFcn
   n::Int
   l::Int
   m::Int
end

function RnYlmBasisFcn(t::VecOrTup)
   @assert length(t) == 3
   return RnYlmBasisFcn(t[1], t[2], t[3])
end

Base.show(io::IO, b::RnYlmBasisFcn) = print(io, "nlm[$(b.n),$(b.l),$(b.m)]")

write_dict(b::RnYlmBasisFcn) =
   Dict("__id__" => "ACE_RnYlmBasisFcn",
        "nlm" => [ b.n, b.l, b.m ] )

read_dict(::Val{:SHIPs_RnYlmBasisFcn}, D::Dict) =
   read_dict(Val{:ACE_RnYlmBasisFcn}(), D)

read_dict(::Val{:ACE_RnYlmBasisFcn}, D::Dict) =
   RnYlmBasisFcn(D["nlmz"]...)

scaling(b::RnYlmBasisFcn, p) = b.n^p + b.l^p + abs(b.m)^p


@doc raw"""
`struct RnYlm1pBasis <: OneParticleBasis`

One-particle basis of the form
```math
\phi_{nlm}({\bm r}) = J_{n}(r) Y_l^m(\hat{\br r})
```
where ``J_{n}`` denotes a radial basis.
"""
struct RnYlm1pBasis{T, TJ <: ScalarBasis{T}} <: OneParticleBasis{T}
   J::TJ           # J = R
   SH::SHBasis{T}  # SH = Ylm
   spec::Vector{RnYlmBasisFcn}
end

function RnYlm1pBasis(J::ScalarBasis{T}, zlist::SZList,
                      spec::Vector{RnYlmBasisFcn}) where {T}
   # now get the maximum L-degree to generate the SH basis
   maxL = maximum(b.l for b in spec)
   SH = SHBasis(maxL, T)
   # construct a preliminary Aindices array to get an incorrect basis
   NZ = length(zlist)
   P = RnYlm1pBasis(J, SH, zlist, spec,
                       Matrix{UnitRange{Int}}(undef, NZ, NZ))
   return P
end

function RnYlm1pBasis(J::ScalarBasis;
                         species = :X,
                         D::AbstractDegree = SparsePSHDegree())
   # get a generic basis spec
   spec = _build_PSH_1p_spec(length(J), D, species)
   # construct the basis
   zlist = ZList(species; static=true)
   return RnYlm1pBasis(J, zlist, spec)
end

cutoff(basis::RnYlm1pBasis) = cutoff(basis.J)

Base.length(basis::RnYlm1pBasis, z0::AtomicNumber) =
      numz(basis) *  length(basis.spec)

Base.length(basis::RnYlm1pBasis, iz0::Integer) =
      numz(basis) *  length(basis.spec)

Base.length(basis::RnYlm1pBasis, iz::Integer, iz0::Integer) =
      length(basis.spec)

Base.rand(basis::RnYlm1pBasis) =
      AtomState(rand(basis.zlist.list), ACE.Random.rand_vec(basis.J))

function get_basis_spec(basis::RnYlm1pBasis, z0::AtomicNumber)
   iz0 = z2i(basis, z0)
   len_iz0 = sum(length(basis, iz, iz0) for iz = 1:numz(basis))
   spec = Vector{RnYlmBasisFcn}(undef, len_iz0)
   for iz = 1:numz(basis)
      z = i2z(basis, iz)
      spec[basis.Aindices[iz, iz0]] =
               [ RnYlmBasisFcn(b.n, b.l, b.m, z)   for b in basis.spec ]
   end
   return spec
end

get_basis_spec(basis::RnYlm1pBasis, iz0::Integer, i::Integer) =
   get_basis_spec(basis, i2z(basis, iz0), i)

function get_basis_spec(basis::RnYlm1pBasis, z0::AtomicNumber, i::Integer)
   iz = ((i-1) ÷ length(basis.spec)) + 1
   inew = mod1(i, length(basis.spec))
   b = basis.spec[inew]
   return RnYlmBasisFcn(b.n, b.l, b.m, i2z(basis, iz))
end


_build_specnl(maxdeg, D, z, z0) =
         gensparse(2, maxdeg;               #     n     l    m  z
                    tup2b = t -> RnYlmBasisFcn(t[1]+1, t[2], 0, z),
                    degfun = b -> D(b, z0),
                    ordered = false)

@doc raw"""
`function _build_PSH_1p_spec`

Construct the specification for a ``P \otimes Y`` type 1-particle basis.
These must be treated differently because of the requirements that complete
``l``-blocks are represented in the basis.

See also: `_get_1p_spec`.
"""
function _build_PSH_1p_spec(maxn::Integer, D::AbstractDegree, species)
   # find out what the largest degree is that we can allow:
   specnl = []
   for s1 in species, s2 in species
      z, z0 = AtomicNumber(s1), AtomicNumber(s2)
      maxdeg = maximum( D(RnYlmBasisFcn(n, 0, 0, z), z0) for n = 1:maxn )
      # generate the `spec::Vector{RnYlmBasisFcn}` using maxn
      specnl_zz0 = _build_specnl(maxdeg, D, z, z0)
      append!(specnl, specnl_zz0)
   end
   specnl = unique([ RnYlmBasisFcn(b.n, b.l, 0, 0) for b in specnl ])
   # add the m-parameters
   return [ RnYlmBasisFcn(b.n, b.l, m, 0)
              for b in specnl for m = -b.l:b.l ]
end

_build_PSH_1p_spec(maxn::Integer, D::AbstractDegree, species::Symbol) =
         _build_PSH_1p_spec(maxn::Integer, D, (species,))

# ------------------------------------------------------
#  FIO code


==(P1::RnYlm1pBasis, P2::RnYlm1pBasis) =  ACE._allfieldsequal(P1, P2)

write_dict(basis::RnYlm1pBasis{T}) where {T} = Dict(
      "__id__" => "ACE_RnYlm1pBasis",
           "J" => write_dict(basis.J),
          "SH" => write_dict(basis.SH),
        "spec" => write_dict.(basis.spec),
       "zlist" => write_dict(basis.zlist),
   )

read_dict(::Val{:SHIPs_RnYlm1pBasis}, D::Dict) =
   read_dict(Val{:ACE_RnYlm1pBasis}(), D::Dict)

function read_dict(::Val{:ACE_RnYlm1pBasis}, D::Dict)
   J = read_dict(D["J"])
   SH = read_dict(D["SH"])
   zlist = read_dict(D["zlist"])
   spec = read_dict.(D["spec"])
   P = RnYlm1pBasis(J, SH, zlist, spec,
                 Matrix{UnitRange{Int}}(undef, length(zlist), length(zlist)))
   set_Aindices!(P)
   return P
end



# ------------------------------------------------------
#  Evaluation code


fltype(basis::RnYlm1pBasis{T}) where T = Complex{T}
rfltype(basis::RnYlm1pBasis{T}) where T = T

alloc_temp(basis::RnYlm1pBasis, args...) =
      (
        BJ = alloc_B(basis.J, args...),
        tmpJ = alloc_temp(basis.J, args...),
        BY = alloc_B(basis.SH, args...),
        tmpY = alloc_temp(basis.SH, args...),
       )

function evaluate!(B, tmp, basis::RnYlm1pBasis,
                   X::AbstractState, X0::AbstractState)
   # evaluate the r-basis and the R̂-basis for the current neighbour at R
   evaluate!(tmp.BJ, tmp.tmpJ, basis.J, norm(X.rr))
   evaluate!(tmp.BY, tmp.tmpY, basis.SH, X.rr)
   # add the contributions to the A_zklm
   @inbounds for (i, nlm) in enumerate(basis.spec)
      B[i] = tmp.BJ[nlm.n] * tmp.BY[index_y(nlm.l, nlm.m)]
   end
   return B
end

# function add_into_A!(A, tmp, basis::RnYlm1pBasis,
#                      R, iz::Integer, iz0::Integer)
#    # evaluate the r-basis and the R̂-basis for the current neighbour at R
#    evaluate!(tmp.BJ, tmp.tmpJ, basis.J, norm(R))
#    evaluate!(tmp.BY, tmp.tmpY, basis.SH, R)
#    # add the contributions to the A_zklm
#    @inbounds for (i, nlm) in enumerate(basis.spec)
#       A[i] += tmp.BJ[nlm.n] * tmp.BY[index_y(nlm.l, nlm.m)]
#    end
#    return nothing
# end


# function add_into_A!(A, inds, tmp, basis::RnYlm1pBasis,
#                      R, iz::Integer, iz0::Integer)
#    # evaluate the r-basis and the R̂-basis for the current neighbour at R
#    evaluate!(tmp.BJ, tmp.tmpJ, basis.J, norm(R))
#    evaluate!(tmp.BY, tmp.tmpY, basis.SH, R)
#    # add the contributions to the A_zklm
#    @inbounds for (i, nlm) in zip(inds, basis.spec)
#       A[i] += tmp.BJ[nlm.n] * tmp.BY[index_y(nlm.l, nlm.m)]
#    end
#    return nothing
# end

alloc_temp_d(basis::RnYlm1pBasis, args...) =
      (
        BJ = alloc_B(basis.J, args...),
        tmpJ = alloc_temp(basis.J, args...),
        BY = alloc_B(basis.SH, args...),
        tmpY = alloc_temp(basis.SH, args...),
        dBJ = alloc_dB(basis.J, args...),
        tmpdJ = alloc_temp_d(basis.J, args...),
        dBY = alloc_dB(basis.SH, args...),
        tmpdY = alloc_temp_d(basis.SH, args...),
       )

function add_into_A_dA!(A, dA, tmpd, basis::RnYlm1pBasis, R, iz::Integer, iz0::Integer)
   r = norm(R)
   R̂ = R / r
   # evaluate the r-basis and the R̂-basis for the current neighbour at R
   evaluate_d!(tmpd.BJ, tmpd.dBJ, tmpd.tmpdJ, basis.J, r)
   evaluate_d!(tmpd.BY, tmpd.dBY, tmpd.tmpdY, basis.SH, R)
   # add the contributions to the A_zklm, ∇A
   @inbounds for (i, nlm) in enumerate(basis.spec)
      iY = index_y(nlm.l, nlm.m)
      A[i] += tmpd.BJ[nlm.n] * tmpd.BY[iY]
      dA[i] = (tmpd.dBJ[nlm.n] * tmpd.BY[iY]) * R̂ + tmpd.BJ[nlm.n] * tmpd.dBY[iY]
   end
   return nothing
end
