
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



import ACE
import Base: ==, convert, Dict, vec
import JuLIP.FIO: read_dict, write_dict

# ---------


"""
`Bond1ParticleFcn` : represents the environment part of a single-particle
basis function.
"""
struct Bond1ParticleFcn{TI}
   kr::TI
   kθ::TI
   kz::TI
end

_inttype(b::Bond1ParticleFcn{TI}) where {TI} = TI

_tuple(b::Bond1ParticleFcn) = (b.kr, b.kθ, b.kz)
Base.vec(b::Bond1ParticleFcn) = [b.kr, b.kθ, b.kz]

function Bond1ParticleFcn(krθz::Union{AbstractVector, Tuple}, INT=Int16)
   @assert length(krθz) == 3
   return Bond1ParticleFcn(INT(krθz[1]), INT(krθz[2]), INT(krθz[3]))
end

Bond1ParticleFcn(b::Bond1ParticleFcn, INT=Int16) =
   Bond1ParticleFcn(INT(b.kr), INT(b.kθ), INT(b.kz))


"""
`BondAList` : datastructure for storing the density projection onto the
1-particle basis functions.
"""
struct BondAList{TI}
   i2krθz::Vector{Bond1ParticleFcn{TI}}
   krθz2i::Dict{Bond1ParticleFcn{TI}, TI}
   # firstz::Vector{TI}
end

_inttype(b::BondAList{TI}) where {TI} = TI
_inttype(b::NTuple{N, TI}) where {N, TI <: Integer} = TI
_inttype(b::StaticArray{DIMS, TI}) where {DIMS, TI <: Integer} = TI



# --------------(de-)serialisation----------------------------------------
write_dict(alist::BondAList{TI}) where {TI} =
      Dict( "__id__" => "ACE_BondAList",
            "TI"     => string(TI),
            "i2krθz" => vec.(alist.i2krθz))

read_dict(::Val{:ACE_BondAList}, D::Dict) =
      BondAList( Bond1ParticleFcn.(D["i2krθz"],
                 Meta.eval(Meta.parse(D["TI"])))  )

==(al1::BondAList, al2::BondAList) = (al1.i2krθz == al2.i2krθz)
# ------------------------------------------------------------------------

Base.length(alist::BondAList) = length(alist.i2krθz)
Base.getindex(alist::BondAList, i::Integer) = alist.i2krθz[i]
Base.getindex(alist::BondAList, krθz::Bond1ParticleFcn) = alist.krθz2i[krθz]

# TODO: rename this to alloc_B ???
alloc_A(alist::BondAList, T=Float64) = zeros(Complex{T}, length(alist))

function BondAList(krθzlist::AbstractVector)
   INT = _inttype(krθzlist[1])
   i2krθz = Bond1ParticleFcn.(krθzlist, INT)
   # create the inverse mapping
   krθz2i = Dict{Bond1ParticleFcn{INT}, INT}()
   for i = 1:length(i2krθz)
      krθz2i[i2krθz[i]] = INT(i)
   end
   # TODO: z dependence
   # # find the first index for each z
   # zmax = maximum( a.z for a in i2krθz )
   # firstz = [ findfirst([a.z == iz for a in i2krθz])
   #            for iz = 1:zmax ]
   return BondAList( i2krθz, krθz2i ) # , [firstz; length(i2krθz)+1] )
end

function BondAList(krθzlist::AbstractVector{<: Bond1ParticleFcn})
   INT = _inttype(krθzlist[1])
   i2krθz = collect(krθzlist)
   # create the inverse mapping
   krθz2i = Dict{Bond1ParticleFcn{INT}, INT}()
   for i = 1:length(i2krθz)
      krθz2i[i2krθz[i]] = INT(i)
   end
   # TODO: z dependence
   # # find the first index for each z
   # zmax = maximum( a.z for a in i2krθz )
   # firstz = [ findfirst([a.z == iz for a in i2krθz])
   #            for iz = 1:zmax ]
   return BondAList( i2krθz, krθz2i ) # , [firstz; length(i2krθz)+1] )
end



# ---------

"""
`BondBasisFcnIdx{N}` : represents a single multivariate basis function
in terms of polynomial degrees in each coordinate direction.
"""
struct BondBasisFcnIdx{N, TI}
   k0::TI
   kkrθz::NTuple{N, Bond1ParticleFcn{TI}}

   function BondBasisFcnIdx(k0::TI1, kkrθz::NTuple{N, Bond1ParticleFcn{TI2}}
            ) where {TI1 <: Integer, N, TI2 <: Integer}
      return new{N, TI2}(TI2(k0), kkrθz)
   end
   function BondBasisFcnIdx(k0::TI1, kkrθz::Tuple{}
            ) where {TI1 <: Integer}
      return new{0, TI1}(TI1(k0), kkrθz)
   end
end

Base.length(b::BondBasisFcnIdx) = length(b.kkrθz)

ACE.bodyorder(b::BondBasisFcnIdx{N}) where {N} = N

BondBasisFcnIdx(k0::Integer, ainds::AbstractVector) =
      BondBasisFcnIdx(k0, tuple(ainds...))

Base.getindex(b::BondBasisFcnIdx, n::Integer) = b.kkrθz[n]

vec(b::BondBasisFcnIdx{N}) where {N} =
   vcat([b.k0], [ vec(b.kkrθz[n]) for n = 1:N ]...)

_tuple(b::BondBasisFcnIdx) = tuple( vec(b)... )
BondBasisFcnIdx(t::Tuple) =
      BondBasisFcnIdx( t[1],
                       [ Bond1ParticleFcn((t[i], t[i+1], t[i+3]))
                         for i = 2:3:length(t) ] )

"""
`BondAAList` : represents a basis functions for an EnvPairPot
"""
mutable struct BondAAList{TI}
   alist::BondAList{TI}
   i2Aidx::Matrix{TI}    # where in A can we find the ith basis function
   i2k0::Vector{TI}      # where in P0 can we find the ith basis function
   len::Vector{TI}       # body-order of ith basis function
   kkrθz2i::Dict{BondBasisFcnIdx, Int}   # inverse mapping
end

kkrθz2i(aalist::BondAAList, kkrθz::BondBasisFcnIdx) = aalist.kkrθz2i[kkrθz]

i2k0(aalist::BondAAList, i) = aalist.i2k0[i]

# dispatch this to a version with the individual arguments so we can also
# use it for de-serialization
i2kkrθz(aalist::BondAAList, i::Integer) =
      i2kkrθz(alist, aalist.i2Aidx, aalist.len, i)

i2kkrθz(alist::BondAList, i2Aidx::AbstractMatrix, len::AbstractVector,
        i::Integer) =
   [ alist.i2krθz[ i2Aidx[i, j] ] for j = 1:len[i] ]

Base.length(aalist::BondAAList) = length(aalist.len)

function BondAAList(atuples, aatuples::AbstractVector{<: NTuple{N}}) where {N}
   INT = _inttype(atuples[1])
   len = length(aatuples)
   alist = BondAList(atuples)
   # assemble the aalist elements
   i2Aidx = zeros(INT, length(aatuples), maximum(length, aatuples))
   len = zeros(INT, length(aatuples))
   kkrθz2i = Dict{BondBasisFcnIdx, Int}()
   for (i, aa) in enumerate(aatuples)
      i2Aidx[i, 1:length(aa)] .= aa
      len[i] = length(aa)
      b1 = BondBasisFcnIdx(0, ntuple(i -> alist[aa[i]], N))
      kkrθz2i[b1] = i
   end
   i2k0 =  zeros(INT, length(aatuples))
   # wrap up...
   return BondAAList(alist, i2Aidx, i2k0, len, kkrθz2i)
end


function BondAAList( Abasis::AbstractVector{<: Bond1ParticleFcn},
                    AAbasis::AbstractVector{<: BondBasisFcnIdx} )
   INT = _inttype(Abasis[1])
   alist = BondAList(Abasis)
   # assemble the aalist elements
   i2Aidx = zeros(INT, length(AAbasis), maximum(length, AAbasis))
   len = zeros(INT, length(AAbasis))
   kkrθz2i = Dict{BondBasisFcnIdx, Int}()
   i2k0 =  zeros(INT, length(AAbasis))
   for (i, AA) in enumerate(AAbasis)
      kkrθz2i[AA] = i
      len[i] = length(AA)
      i2k0[i] = AA.k0
      for j = 1:length(AA)
         i2Aidx[i, j] = alist.krθz2i[AA.kkrθz[j]]
      end
   end
   # wrap up...
   return BondAAList(alist, i2Aidx, i2k0, len, kkrθz2i)

end

# --------------(de-)serialisation----------------------------------------

write_dict(aalist::BondAAList) =
      Dict( "__id__" => "ACE_BondAAList",
            "alist"  => write_dict(aalist.alist),
            "i2Aidx" => write_dict(aalist.i2Aidx),
            "i2k0"   => aalist.i2k0,
            "len"    => aalist.len )

function read_dict(::Val{:ACE_BondAAList}, D::Dict)
   alist = read_dict(D["alist"])
   TI = _inttype(alist)
   i2Aidx = read_dict(D["i2Aidx"])::Matrix{TI}
   i2k0 = convert(Vector{TI}, D["i2k0"])
   len = convert(Vector{TI}, D["len"])
   kkrθz2i = Dict{BondBasisFcnIdx, Int}()
   for i = 1:length(len)
      kkrθz = i2kkrθz(alist, i2Aidx, len, i)
      bfcn = BondBasisFcnIdx(i2k0[i], kkrθz)
      kkrθz2i[bfcn] = Int(i)
   end
   return BondAAList(alist, i2Aidx, i2k0, len, kkrθz2i)
end

==(aal1::BondAAList, aal2::BondAAList) = (
         (aal1.i2Aidx == aal2.i2Aidx) &&
         (aal1.i2k0 == aal2.i2k0)     &&
         (aal1.alist == aal2.alist)
      )
