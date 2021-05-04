
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using ACE: PolyTransform,
             IdTransform,
             TransformedJacobi,
             PolyCutoff1s,
             PolyCutoff2s,
             gensparse

# ------ Basis generation

totaldegree(b::Bond1ParticleFcn, wr, wθ, wz) =
            wr * b.kr + wθ * abs(b.kθ) + wz * b.kz

totaldegree(b::BondBasisFcnIdx, wr, wθ, wz) =
            b.k0 + sum(totaldegree(b1, wr, wθ, wz) for b1 in b.kkrθz)

totaldegree(b::BondBasisFcnIdx{0}, wr, wθ, wz) =
            b.k0

"""
envpairbasis(species, N;
             rcut0 = nothing,
             degree = nothing,
             rnn = species == :X ? 2.7 : JuLIP.rnn(species),
             rcutenvfact = 1.9,
             rinenvfact = 0.66,
             rinr0 = rcutenvfact * rnn,
             rcutr = rcutenvfact * rnn,
             rinr = 0.0,
             rcutz = rcutenvfact * rnn + 0.5 * rcut0,
             wenv = 4.0, wr = 1.0, wz = 1.0, wθ = 1.0,
             r0trans = PolyTransform(2, rnn),
             rtrans = PolyTransform(2, rnn),
             ztrans = IdTransform(),
             r0fcut = PolyCutoff1s(2, rinr0, rcut0),
             rfcut = PolyCutoff1s(2, 0.0, rcutr),
             zfcut = PolyCutoff2s(2, -rcutz, rcutz)
             )
"""
envpairbasis(species, N, args...; kwargs...) =
      envpairbasis(species, Val(N), args...; kwargs...)

function  envpairbasis(species, ::Val{N};
             rcut0 = nothing,
             degree = nothing,
             rnn = species == :X ? 2.7 : JuLIP.rnn(species),
             rcutenvfact = 1.9,
             rinenvfact = 0.66,
             rinr0 = rcutenvfact * rnn,
             rcutr = rcutenvfact * rnn,
             rinr = 0.0,
             rcutz = rcutenvfact * rnn + 0.5 * rcut0,
             wenv = 4.0, wr = 1.0, wz = 1.0, wθ = 1.0,
             r0trans = PolyTransform(2, rnn),
             rtrans = IdTransform(),
             ztrans = IdTransform(),
             r0fcut = PolyCutoff1s(2, rinr0, rcut0),
             rfcut = PolyCutoff1s(2, 0.0, rcutr),
             zfcut = PolyCutoff2s(2, -rcutz, rcutz),
             zsymm = true,
             ) where {N}

   # some basic health checks.
   @assert (degree isa Integer) && (degree > 0)
   @assert (rcut0 isa Real) && (rcut0 > 0)
   @assert (wr >= 1) && (wz >= 1) && (wθ >= 1)

   degenv = degree / wenv

   # put together the basis specification
   # -------------------------------------
   # first generate a list of 1-particle functions
   degfunA = t -> wr * t[1] + wθ * abs(cyl_i2l(t[2]+1)) + wz * t[3]
   atuples = gensparse(3; ordered = false,
                          admissible = t -> (degfunA(t) <= degenv + 1))
   sort!(atuples; by = degfunA)
   Abasis = map( t -> Bond1ParticleFcn((t[1], cyl_i2l(t[2]+1), t[3])), atuples )

   # now to generate products of As we take N-tuples
   #     t = (t0, t1, ..., tN)  where ti is an index pointing into Abasis
   function aabfcn(t)
      if isempty(t)
         tnz = t
      else
         tnz = t[findall(t .!= 0)] # tuple
      end
      return BondBasisFcnIdx(0, Abasis[[tnz...]])
   end
   degreefunenv = t -> totaldegree(aabfcn(t), wr, wθ, wz)
   aatuples = gensparse(N; ordered = true,
                           admissible = t -> (degreefunenv(t) <= degenv))

   # -------------
   # Filtering ...
   # -------------
   AAbasis = BondBasisFcnIdx[]
   for aa in aatuples
      AA = aabfcn(aa)
      if length(AA) == 0
         sumkθ = 0
         sumkz = 0
      else
         sumkθ = sum( A.kθ for A in AA.kkrθz )
         sumkz = sum( A.kz for A in AA.kkrθz )
      end
      if sumkθ == 0 && (iseven(sumkz) || !zsymm)
         push!(AAbasis, AA)
      end
   end

   # -------------------------
   # Combine with P0 basis ...
   # -------------------------
   k0AAbasis = BondBasisFcnIdx[]
   for k0 = 0:degree, iAA = 1:length(AAbasis)
      AA = AAbasis[iAA]
      if k0 + wenv * totaldegree(AA, wr, wθ, wz) <= degree
         push!(k0AAbasis, BondBasisFcnIdx(k0, AA.kkrθz))
      end
   end

   # Now generate the aalist datastructure
   aalist = BondAAList(Abasis, k0AAbasis)

   # generate the scalar polynomials
   deg0 = degree
   P0 = TransformedJacobi(deg0, r0trans, r0fcut)
   degr = maximum(t -> t[1], atuples)
   Pr = TransformedJacobi(degr, rtrans, rfcut)
   degθ = maximum(t -> t[2], atuples)
   Pθ = FourierBasis(degθ, Float64)
   degz = maximum(t -> t[3], atuples)
   Pz = TransformedJacobi(degz, ztrans, zfcut)

   # put together the basis
   return EnvPairBasis(P0, Pr, Pθ, Pz, aalist)
end
