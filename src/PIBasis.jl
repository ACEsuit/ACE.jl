
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


import SHIPs: alloc_B, alloc_dB
import Base: ==
import JuLIP: evaluate!, evaluate_d!,
              alloc_temp, alloc_temp_d,
              read_dict, write_dict

import JuLIP.MLIPs: IPBasis

struct PIBasis{T, NZ, TR} <: IPBasis
   J::TR                    # specifies the radial basis  / n = kr
   SH::SHBasis{T}
   zlist::SZList{NZ}
   # -------------- A and AA datastructures + coefficients
   alists::NTuple{NZ, AList}
   aalists::NTuple{NZ, AAList}
end


cutoff(pib::PIBasis) = cutoff(pib.J)

==(S1::PIBasis, S2::PIBasis) =
      all( getfield(S1, i) == getfield(S2, i)
           for i = 1:fieldcount(PIBasis) )

Base.length(pib::PIBasis) = sum(length.(pib.aalists))

order(pib::PIBasis{T, NZ}) where {T, NZ} = NZ


# ------------------------------------------------------------
#   FIO code
# ------------------------------------------------------------

write_dict(pib::PIBasis{T,NZ}) where {T, NZ} = Dict(
      "__id__" => "SHIPs_PIBasis",
      "Pr" => Dict(pib.J),
      "SH_maxL" => pib.SH.maxL,   # TODO: replace this with Dict(SH)
      "T" => string(eltype(pib.SH)),
      "zlist" => Dict(pib.zlist),
      "alists" => [Dict.(pib.alists)...],
      "aalists" => [Dict.(pib.aalists)...],
   )

read_dict(::Val{:SHIPs_PIBasis}, D::Dict) = PIBasis(D)

# bodyorder - 1 is because BO is the number of neighbours
# not the actual body-order
function PIBasis(D::Dict)
   T = Meta.eval(Meta.parse(D["T"]))
   Pr = TransformedJacobi(D["Pr"])
   SH = SHBasis(D["SH_maxL"], T)
   zlist = decode_dict(D["zlist"])
   NZ = length(zlist)
   alists = ntuple(i -> AList(D["alists"][i]), NZ)
   aalists = ntuple(i -> AAList(D["aalists"][i], alists[i]), NZ)
   return  PIBasis(Pr, SH, zlist, alists, aalists)
end


# ------------------------------------------------------------
#   Initialisation code
# ------------------------------------------------------------

# - leverage the generic sparse thing
# - filtering moved to afterwards...


function PIBasis(N, Pr;
                 totaldegree = nothing, wn = 1.0, wl = 2.0, wm = 2.0)

   # t[1] = n
   # t[2] = l
   # t[3] -> index into [-l, ..., l], starting with 0, so
   #         -l + t[3] == m
   degfunA = t -> wn * t[1] + wl * t[2] + wm * abs(-t[2]+t[3])
   atuples = gensparse(3; ordered = false,
                          admissible = t -> (degfunA(t) <= totaldegree+1))
   sort!(atuples; by = degfunA)

   # now to construct the AA basis, a tuple
   #  t = (t1, ..., tN)
   # indexes into `atuples` with a 0-index denoting a reduction in
   # interaction order.
   degfunA_ = ti -> ti == 0 ? 0 : degfunA(atuples[ti])
   degfunAA = t -> sum(degfunA_, t)
   aatuples = gensparse(N; ordered = true,
                           admissible = t -> (degfunAA(t) <= totaldegree))

   # next we need to get rid of the (0,...,0) tuple, which corresponds to
   # a constant function; TODO: allow this! this is just the case N = 0!!!!
   I0 = findall(sum.(aatuples) .== 0)
   @assert length(I0) == 1
   aatuples = deleteat!(aatuples, I0[1])

   # ----------------------------------------------------
   #   here we could insert some filtering mechanisms!
   # ----------------------------------------------------

   # construct the alist::AList
   zklmtuples = [ (z=Int16(1), k=IntS(t[1]), l=IntS(t[2]), m=IntS(-t[2]+t[3]))
                   for t in atuples ]
   alist = AList(zklmtuples)
   maxL = maximum( zklm.l for zklm in zklmtuples )

   # construct the aalist::AAList
   # construct a (zz =, kk =, ll =, mm = ) named tuple from a t ∈ aatuples
   # this will at the same time "shorten" the tuples to throw away the zeros
   function aat2ZKLM(t)
      tnz = t[ findall(t .!= 0) ]  # throw away the zeros
      zklms = zklmtuples[tnz]      # get the (z=, k=, l=, m=) tuples
      return (zz = SVector([s.z for s in zklms]...),
              kk = SVector([s.k for s in zklms]...),
              ll = SVector([s.l for s in zklms]...),
              mm = SVector([s.m for s in zklms]...))
   end
   ZKLM_list = aat2ZKLM.(aatuples)
   aalist = AAList(ZKLM_list, alist)

   return PIBasis( Pr, SHBasis(maxL), JuLIP.Potentials.ZList(:X, static=true),
                   (alist,), (aalist,) )
end


# ------------------------------------------------------------
#   Evaluation code
# ------------------------------------------------------------

alloc_B(pib::PIBasis{T,NZ}, args...) where {T, NZ} =
   zeros(Complex{T}, length(pib))

alloc_temp(pib::PIBasis{T,NZ}, args...) where {T, NZ} =
   (     J = alloc_B(pib.J),
         Y = alloc_B(pib.SH),
         A = [ alloc_A(pib.alists[iz0])  for iz0 = 1:NZ ],
      tmpJ = alloc_temp(pib.J),
      tmpY = alloc_temp(pib.SH)
           )

function _get_I_iz0(pib::PIBasis, iz0)
   I0 = (iz0 == 1) ? 0 : sum( length(pib.aalists[iz])  for iz = 1:(iz0-1) )
   return I0 .+ (1:length(pib.aalists[iz0]))
end

# compute one site energy
function evaluate!(B, tmp, pib::PIBasis{T},
                   Rs::AbstractVector{JVec{T}},
                   Zs::AbstractVector{<:Integer},
                   z0::Integer) where {T}
   iz0 = z2i(pib, z0)
   A, alist, aalist = tmp.A[iz0], pib.alists[iz0], pib.aalists[iz0]
   Iz0 = _get_I_iz0(pib, iz0)
   precompute_A!(A, tmp, alist, Rs, Zs, pib)
   fill!(B, 0)
   for i = 1:length(aalist)
      AAi = one(Complex{T})
      for α = 1:aalist.len[i]
         AAi *= A[aalist.i2Aidx[i, α]]
      end
      B[Iz0[i]] = AAi
   end
   return B
end
