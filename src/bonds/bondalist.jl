
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------




import Base: ==, convert, Dict

# ---------

"""
`BondBasisFcnIdx{N}` : represents a single multivariate basis function
in terms of polynomial degrees in each coordinate direction.
"""
struct BondBasisFcnIdx{N}
   m0::Int16
   kk::SVector{N, Int16}  # r-coord
   ll::SVector{N, Int16}  # θ-coord
   mm::SVector{N, Int16}  # z-coord
end

PoSH.bodyorder(b::BondBasisFcnIdx{N}) where {N} = N

"""
`Bond1ParticleFcn` : represents the environment part of a single-particle
basis function.
"""
struct Bond1ParticleFcn
   k::Int16
   l::Int16
   m::Int16
end

Base.getindex(b::BondBasisFcnIdx, n::Integer) =
            Bond1ParticleFcn(b.kk[n], b.ll[n], b.mm[n])


"""
`AList` : datastructure to help compute the A_zklm density projections

* `i2zklm` : list of all admissible `(z,k,l,m)` tuples
* `zklm2i` : dictionary with (z,k,l,m) keys to compute  the map `(z,k,l,m) -> i`
* `firstz` : `firstz[iz]` stores the first index in the A_zklm array for with
             z = zi. This can be used to iterate over all A entries for which
             z = zi. (they are sorted by z first)

TODO: consider switching to Int8 and Int16 indexing
"""
struct BondAList
   i2zklm::Vector{zklmTuple}
   zklm2i::Dict{zklmTuple, IntS}
   firstz::Vector{IntS}
end

# --------------(de-)serialisation----------------------------------------
Dict(alist::AList) = Dict("__id__" => "PoSH_AList",
                           "i2zklm" => zklm2vec.(alist.i2zklm))
AList(D::Dict) = AList(vec2zklm.(D["i2zklm"]))
zklm2vec(t) = [t.z, t.k, t.l, t.m]
vec2zklm(v) = (z=Int16(v[1]), k=IntS(v[2]), l=IntS(v[3]), m=IntS(v[4]))
==(al1::AList, al2::AList) = (al1.i2zklm == al2.i2zklm)
# ------------------------------------------------------------------------

Base.length(alist::AList) = length(alist.i2zklm)
Base.getindex(alist::AList, i::Integer) = alist.i2zklm[i]
Base.getindex(alist::AList, zklm::zklmTuple) = alist.zklm2i[zklm]

alloc_A(alist::AList, T=Float64) = zeros(Complex{T}, length(alist))

function AList(zklmlist::AbstractVector{zklmTuple})
   # sort the tuples - by z, then k, then l, then m
   i2zklm = sort(zklmlist)
   # create the inverse mapping
   zklm2i = Dict{zklmTuple, IntS}()
   for i = 1:length(i2zklm)
      zklm2i[i2zklm[i]] = IntS(i)
   end
   # find the first index for each z
   zmax = maximum( a.z for a in i2zklm )
   firstz = [ findfirst([a.z == iz for a in i2zklm])
              for iz = 1:zmax ]
   return AList( i2zklm, zklm2i, [firstz; length(i2zklm)+1] )
end






# ---------

"""
`BondAAList` : represents the basis functions for an EnvPairPot
"""
mutable struct BondAAList
   i2Aidx::Matrix{IntS}    # where in A can we find the ith basis function
   i2m0::Vector{IntS}      # where in P0 can we find the ith basis function
   len::Vector{IntS}       # body-order of ith basis function
   zklm2i::Dict{BondBasisFcnIdx, IntS}   # inverse mapping
end

# --------------(de-)serialisation----------------------------------------


function Dict(aalist::AAList)
   ZKLM_list = Vector{Any}(undef, length(aalist))
   for (zzkkllmm, i) in aalist.zklm2i
      ZKLM_list[i] = zzkkllmm
   end
   return Dict("__id__" => "PoSH_AAList", "ZKLM_list" => ZKLM_list)
end
zzkkllmm2vec(zzkkllmm) = Vector.([zzkkllmm...])
vec2zzkkllmm(v) = (SVector(Int16.(v[1])...),
                   SVector(IntS.(v[2])...),
                   SVector(IntS.(v[3])...),
                   SVector(IntS.(v[4])...))
AAList(D::Dict, alist) = AAList(vec2zzkkllmm.(D["ZKLM_list"]), alist)
==(aal1::AAList, aal2::AAList) = (aal1.i2Aidx == aal2.i2Aidx)
# ------------------------------------------------------------------------


bodyorder(aalist::AAList) = maximum(aalist.len) + 1

Base.length(aalist::AAList) = length(aalist.len)

Base.getindex(aalist::AAList, t::Tuple) = aalist.zklm2i[t]

alloc_AA(aalist::AAList, T = Float64) = zeros(Complex{T}, length(aalist))

"""
`AAList(ZKLM_list, alist)` : standard AAList constructor,
* `ZKLM_list` : a collection of (izz=?, kk=?, ll=?, mm=?) tuples
* `alist` a compatible `AList`
"""
function AAList(ZKLM_list, alist)
   BO = maximum(ν -> length(ν[1]), ZKLM_list)  # body-order -> size of iAidx

   # create arrays to construct AAList
   aalist = AAList(Matrix{IntS}(undef,0,BO), IntS[], Dict{Any, IntS}())

   for (izz, kk, ll, mm) in ZKLM_list
      push!(aalist, (izz, kk, ll, mm), alist)
   end
   return aalist
end

function Base.push!(aalist::AAList, tpl, alist)
   izz, kk, ll, mm = tpl
   BO = size(aalist.i2Aidx, 2)
   # store in the index of the current row in the reverse map
   idx = length(aalist) + 1
   aalist.zklm2i[(SVector(izz...), SVector(kk...), SVector(ll...), SVector(mm...))] = idx
   # store the body-order of the current ∏A function
   push!(aalist.len, length(ll))

   # fill the row of the i2Aidx matrix
   newrow = IntS[]
   for α = 1:length(ll)
      zklm = (z=izz[α], k=kk[α], l=ll[α], m=IntS(mm[α]))
      push!(newrow, alist[zklm])
   end
   # fill up the iAidx vector with zeros up to the body-order
   # this will create 0 entries in the matrix after reshaping
   for α = (length(ll)+1):BO
      push!(newrow, 0)
   end
   aalist.i2Aidx = vcat(aalist.i2Aidx, newrow')
   @assert idx == size(aalist.i2Aidx, 1)
   return aalist
end


function precompute_AA!(AA, A, aalist) # tmp, ship::SHIPBasis{T}, iz0) where {T}
   fill!(AA, 1)
   for i = 1:length(aalist)
      for α = 1:aalist.len[i]
         iA = aalist.i2Aidx[i, α]
         AA[i] *= A[iA]
      end
   end
   return nothing
end


# --------------------------------------------------------
# this section of functions is for computing
#  ∂∏A / ∂R_j

# function grad_phi_Rj(Rj, j, zklm, tmp)
#    ik = zklm.k + 1
#    iy = index_y(zklm.l, zklm.m)
#    return ( tmp.dJJ[j, ik] *  tmp.YY[j, iy] * (Rj/norm(Rj))
#            + tmp.JJ[j, ik] * tmp.dYY[j, iy] )
# end

function grad_AA_Ab(iAA, b, alist, aalist, A)
   g = one(eltype(A))
   for a = 1:aalist.len[iAA]
      if a != b
         iA = aalist.i2Aidx[iAA, a]
         g *= A[iA]
      end
   end
   return g
end

function grad_AAi_Rj(iAA, j, Rj::JVec{T}, izj::Integer,
                     alist, aalist, A, AA, dA, tmp) where {T}
   g = zero(JVec{Complex{T}})
   for b = 1:aalist.len[iAA] # body-order
      # A_{n_b} = A[iA]
      iA = aalist.i2Aidx[iAA, b]
      # daa_dab = ∂(∏A_{n_a}) / ∂A_{n_b}
      daa_dab = grad_AA_Ab(iAA, b, alist, aalist, A)
      zklm = alist[iA] # (zklm corresponding to A_{n_b})
      ∇ϕ_zklm = dA[j, iA] # grad_phi_Rj(Rj, j, zklm, tmp)
      g += daa_dab * ∇ϕ_zklm
   end
   return g
end

function grad_AA_Rj!(tmp, ship, j, Rs, Zs, iz0) where {T}
   for iAA = 1:length(ship.aalists[iz0])
      # g = ∂(∏_a A_a) / ∂Rj
      tmp.dAAj[iz0][iAA] = grad_AAi_Rj(iAA, j, Rs[j], z2i(ship, Zs[j]),
                                       ship.alists[iz0], ship.aalists[iz0],
                                       tmp.A[iz0], tmp.AA[iz0], tmp.dA[iz0],
                                       tmp)
   end
   return tmp.dAAj[iz0]
end


# --------------------------------------------------------

"""
convert the "old" `(NuZ, ZKL)` format into the simpler (zz, kk, ll, mm)
format, and at the same time extract the one-particle basis (z, k, l, m)
"""
function alists_from_bgrps(bgrps::Tuple)
   NZ = length(bgrps)
   zzkkllmm_list = [ Tuple[] for _=1:NZ ]
   zklm_set = [ Set() for _=1:NZ ]
   for iz0 = 1:NZ
      for (izz, kk, ll) in bgrps[iz0], mm in _mrange(ll)
         push!(zzkkllmm_list[iz0], (izz, kk, ll, IntS.(mm)))
         for α = 1:length(ll)
            zklm = (z=izz[α], k=kk[α], l=ll[α], m=IntS(mm[α]))
            push!(zklm_set[iz0], zklm)
         end
      end
   end

   alist =  ntuple(iz0 -> AList([ zklm for zklm in collect(zklm_set[iz0]) ]), NZ)
   aalist = ntuple(iz0 -> AAList(zzkkllmm_list[iz0], alist[iz0]), NZ)
   return alist, aalist
end


# --------------------------------------------------------

using SparseArrays: SparseMatrixCSC

function _my_mul!(C::AbstractVector, A::SparseMatrixCSC, B::AbstractVector)
   A.n == length(B) || throw(DimensionMismatch())
   A.m == length(C) || throw(DimensionMismatch())
   nzv = A.nzval
   rv = A.rowval
   fill!(C, zero(eltype(C)))
   @inbounds for col = 1:A.n
      b = B[col]
      for j = A.colptr[col]:(A.colptr[col + 1] - 1)
         C[rv[j]] += nzv[j] * b
      end
   end
   return C
end
