
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



import Base: ==, convert, Dict

const zklmTuple = NamedTuple{(:z, :k, :l, :m), Tuple{Int16, IntS, IntS, IntS}}

"""
`AList` : datastructure to help compute the A_zklm density projections

* `i2zklm` : list of all admissible `(z,k,l,m)` tuples
* `zklm2i` : dictionary with (z,k,l,m) keys to compute  the map `(z,k,l,m) -> i`
* `firstz` : `firstz[iz]` stores the first index in the A_zklm array for with
             z = zi. This can be used to iterate over all A entries for which
             z = zi. (they are sorted by z first)
"""
struct AList
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


"""
This fills the A-array stored in tmp with the A_zklm density projections in
the order specified by AList. It also evaluates the radial and angular basis
functions along the way.
"""
function precompute_A!(A, tmp, alist, Rs, Zs, ship)
   fill!(A, 0)
   for (R, Z) in zip(Rs, Zs)
      # evaluate the r-basis and the R̂-basis for the current neighbour at R
      evaluate!(tmp.J, tmp.tmpJ, ship.J, norm(R))
      evaluate!(tmp.Y, tmp.tmpY, ship.SH, R)
      # add the contributions to the A_zklm
      iz = z2i(ship, Z)
      for i = alist.firstz[iz]:(alist.firstz[iz+1]-1)
         zklm = alist[i]
         A[i] += tmp.J[zklm.k+1] * tmp.Y[index_y(zklm.l, zklm.m)]
      end
   end
   return A
end




function precompute_dA!(A, dA, tmp, alist, Rs, Zs, ship)
   fill!(A, 0)
   fill!(dA, zero(eltype(dA)))
   for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
      # precompute the derivatives of the Jacobi polynomials and Ylms
      evaluate_d!(tmp.J, tmp.dJ, tmp.tmpJ, ship.J, norm(R))
      evaluate_d!(tmp.Y, tmp.dY, tmp.tmpY, ship.SH, R)
      # deduce the A and dA values
      iz = z2i(ship, Z)
      R̂ = R / norm(R)
      for i = alist.firstz[iz]:(alist.firstz[iz+1]-1)
         zklm = alist[i]
         ik = zklm.k+1; iy = index_y(zklm.l, zklm.m)
         A[i] += tmp.J[ik] * tmp.Y[iy]
         # and into dA # grad_phi_Rj(R, iR, zklm, tmp)
         ∇ϕ_zklm = tmp.dJ[ik] * tmp.Y[iy] * R̂ + tmp.J[ik] * tmp.dY[iy]
         dA[iR, i] = ∇ϕ_zklm
      end
   end
   return dA
end

# ---------


"""
`AAList` : datastructure to help compute the A_𝐳𝐤𝐥𝐦 = ∏ A_zklm

* `i2Aidx` : indices in AList of the zklms to avoid the Dict lookup
* `len`    : len[i] is the number of relevant entries of i2zklm[i,:]
             i.e. the body-order of this entry
* `zklm2i` : dictionary of all (z,k,l,m) tuples to compute  the
             map `(zz,kk,ll,mm) -> i`
* `firstz` : `firstz[iz]` stores the first index in the A_zklm array for with
             z = zi. This can be used to iterate over all A entries for which
             z = zi. (they are sorted by z first)
"""
struct AAList
   i2Aidx::Matrix{IntS}    # where in A can we find these
   len::Vector{IntS}       # body-order
   zklm2i::Dict{Any, IntS} # inverse mapping
end

# --------------(de-)serialisation----------------------------------------

==
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
   iAidx = IntS[]
   len = IntS[]
   zklm2i = Dict{Any, IntS}()

   idx = 0
   for (izz, kk, ll, mm) in ZKLM_list
      # store in the index of the current row in the reverse map
      idx += 1
      zklm2i[(izz, kk, ll, mm)] = idx
      # store the body-order of the current ∏A function
      push!(len, length(ll))

      # fill the row of the i2Aidx matrix
      for α = 1:length(ll)
         zklm = (z=izz[α], k=kk[α], l=ll[α], m=IntS(mm[α]))
         iA = alist[zklm]
         push!(iAidx, iA)
      end
      # fill up the iAidx vector with zeros up to the body-order
      # this will create 0 entries in the matrix after reshaping
      for α = (length(ll)+1):BO
         push!(iAidx, 0)
      end
   end
   return AAList( reshape(iAidx, (BO, idx))', len, zklm2i )
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
