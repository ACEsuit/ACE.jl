
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs: PolyTransform,
             IdTransform,
             TransformedJacobi,
             PolyCutoff1s,
             PolyCutoff2s

# ------ Basis generation

totaldegree(b::Bond1ParticleFcn, wr, wθ, wz) =
            wr * b.kr + wθ * abs(b.kθ) + wz * b.kz

totaldegree(b::BondBasisFcnIdx, wr, wθ, wz) =
            b.k0 + sum(totaldegree(b1, wr, wθ, wz) for b1 in b.kkrθz)

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
             rtrans = PolyTransform(2, rnn),
             ztrans = IdTransform(),
             r0fcut = PolyCutoff1s(2, rinr0, rcut0),
             rfcut = PolyCutoff1s(2, 0.0, rcutr),
             zfcut = PolyCutoff2s(2, -rcutz, rcutz)
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
   aabfcn = t -> BondBasisFcnIdx(0, ntuple(i -> Abasis[t[i]+1], N))
   degreefunenv = t -> totaldegree(aabfcn(t), wr, wθ, wz)
   aatuples = gensparse(N; ordered = true,
                           admissible = t -> (degreefunenv(t) <= degenv))

   # -------------
   # Filtering ...
   # -------------
   AAbasis = BondBasisFcnIdx[]
   for aa in aatuples
      AA = aabfcn(aa)
      sumkθ = sum( A.kθ for A in AA.kkrθz )
      sumkz = sum( A.kz for A in AA.kkrθz )
      if sumkθ == 0 && iseven(sumkz)
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



## ----------- Some general utility functions that we should move elsewhere...

# Auxiliary functions to generate sparse grid type stuff

# gensparse(N::Integer, deg, degfun, filter = _->true, TI = Int16) =
#       gensparse(N, ν -> ((degfun(ν) <= deg) && filter(ν)), TI)



gensparse(N::Integer, deg::Integer; degfun = ν -> sum(ν), kwargs...) =
   gensparse(N; admissible = (degfun(ν) <= deg), kwargs...)

gensparse(N::Integer;
          admissible = _-> false,
          filter = _-> true,
          INT = Int16,
          ordered = false) =
      _gensparse(Val(N), admissible, filter, INT, ordered)

function _gensparse(::Val{N}, admissible, filter, INT, ordered) where {N}
   @assert INT <: Integer
   @assert N > 0

   lastidx = 0
   ν = @MVector zeros(INT, N)
   Nu = SVector{N, INT}[]

   while true
      # check whether the current ν tuple is admissible
      # the first condition is that its max index is small enough
      # we want to increment `curindex`, but if we've reach the maximum degree
      # then we need to move to the next index down
      if admissible(ν)
         # ... then we add it to the stack  ...
         # (unless some filtering mechanism prevents it)
         if filter(ν)
            push!(Nu, SVector(ν))
         end
         # ... and increment it
         lastidx = N
         ν[lastidx] += 1
      else
         # we have overshot, e.g. degfun(ν) > deg; we must go back down, by
         # decreasing the index at which we increment
         if lastidx == 1
            # if we have gone all the way down to lastindex==1 and are still
            # inadmissible then this means we are done
            break
         end
         # reset
         ν[lastidx-1] += 1
         if ordered   #   ordered tuples (permutation symmetry)
            ν[lastidx:end] .= ν[lastidx-1]
         else         # unordered tuples (no permutation symmetry)
            ν[lastidx:end] .= 0
         end
         lastidx -= 1
      end
   end

   return Nu
end
