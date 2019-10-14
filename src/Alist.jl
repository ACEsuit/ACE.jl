
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



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

Base.length(alist::AList) = length(alist.i2zklm)
Base.getindex(alist::AList, i::Integer) = alist.i2zklm[i]
Base.getindex(alist::AList, zklm::zklmTuple) = alist.zklm2i[zklm]

_alloc_A(alist::AList, T=Float64) = zeros(Complex{T}, length(alist))
_alloc_dA(alist::AList, T=Float64) = zeros(JVec{Complex{T}}, length(alist))

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
function precompute_A!(tmp, AList, ship::SHIPBasis{BO,T}, Rs, Zs) where {BO, T}
   fill!(zero(T), tmp.A)
   for (R, Z) in zip(Rs, Zs)
      iz = z2i(ship, Z)
      # evaluate the r-basis and the R̂-basis for the current neighbour at R
      eval_basis!(tmp.J, tmp.tmpJ, ship.J, norm(R))
      eval_basis!(tmp.Y, tmp.tmpY, ship.SH, R)
      # add the contributions to the A_zklm
      for i = AList.firstz[iz]:AList.firstz[iz+1]
         zklm = AList[i]
         tmp.A[i] += tmp.J[zklm.k+1] * tmp.Y[index_y(zklm.l, zklm.m)]
      end
   end
   return nothing
end


# ---------

const zzkkllmmTuple{N} = SVector{N, zklmTuple}

"""
`AAList` : datastructure to help compute the A_𝐳𝐤𝐥𝐦 = ∏ A_zklm

* `i2Aidx` : indices in AList of the zklms to avoid the Dict lookup
* `len`    : len[i] is the number of relevant entries of i2zklm[i,:]
             i.e. the body-order of this entry
* `zklm2i` : dictionary of all (z,k,l,m) tuples to compute  the
             map `(z,k,l,m) -> i`
* `firstz` : `firstz[iz]` stores the first index in the A_zklm array for with
             z = zi. This can be used to iterate over all A entries for which
             z = zi. (they are sorted by z first)
"""
struct AAList
   i2Aidx::Matrix{IntS}
   len::Vector{IntS}
   zklm2i::Dict{Any, IntS}
end

Base.length(aalist::AAList) = length(aalist.len)

alloc_AA(aalist::AAList, T = Float64) = zeros(T, length(aalist))

function AAList(NuZ::AbstractVector, ZKL)
   iAidx = IntS[]
   len = IntS[]
   zklm2i = Dict{Any, IntS}()
   idx = 0

   for νz in NuZ
      # get zz, kk, ll
      izz = νz.izz
      kk, ll = _kl(νz.ν, izz, ZKL)
      # loop over the compatible mm
      for mm in _mrange(ll)
         # fill the row of the i2Aidx matrix 
         for α = 1:length(ll)
            zklm = (z=iz, k=kk[α], l=ll[α], m=mm[α])
            push!(iAidx, AList[zklm])
            zzkkllmm[α] = zklm
         end
         # fill up the iAidx vector with zeros up to the body-order
         # this will create 0 entries in the matrix after reshaping
         for α = (length(ll)+1):BO
            push!(iAidx, 0)
         end
         # store in the index of the current row in the reverse map
         idx += 1
         zklm2i[(zz, kk, ll, mm)] = idx
         # store the body-order of the current ∏A function
         push!(len, length(ll))
      end
   end
   return AAList(reshape(i2Aidx, (idx, BO)), len, zklm2i)
end


function precompute_AA!(tmp, aalist, alist, ship::SHIPBasis{BO,T}) where {BO, T}
   fill!(one(Complex{T}), tmp.AA)
   for i = 1:length(aalist)
      for α = 1:aalist.len[i]
         iA = aalist.i2Aidx[i, α]
         tmp.AA[i] *= tmp.A[iA]
      end
   end
   return nothing
end
