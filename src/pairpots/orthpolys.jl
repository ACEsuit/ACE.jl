
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module OrthPolys

using SparseArrays
import PoSH: DistanceTransform,
            transform, transform_d

struct AffineTransform{T, TT} <: DistanceTransform
   trans::TT
   rin::T
   rcut::T
   tin::T
   tcut::T
end

AffineTransform(trans, rin, rcut) =
      AffineTransform(trans, rin, rcut,
                      transform(trans, rin), transform(trans, rcut))

function transform(trans::AffineTransform, r)
   t = transform(trans.trans, r)
   return ( (t - trans.tin)  / (trans.tcut - t.tin)
          - (t - trans.tcut) / (trans.tin - trans.tcut) )
end

function transform_d(trans::AffineTransform, r)
   dt = transform_d(trans.trans, r)
   return (2 / (trans.tcut - t.tin)) * dt
end


struct TransPolyBasis{T, TT} <: IPBasis
   rdf::Vector{T}
   # ----------------- the main polynomial parameters
   A::Vector{T}
   B::Vector{T}
   C::Vector{T}
   T1::SparseMatrixCSC{T,Int}
   # -----------------
   trans::TT
   tin::T
   tcut::T
   # ----------------- parameters to optimise the basis
   # T2::Matrix{T}
   # optim::Bool
end



function TransPolyBasis(N::Integer,
                        trans::DistanceTransform,
                        rin::AbstractFloat,
                        rcut::AbstractFloat,
                        pcut::Integer,
                        rdf::Vector,
                        ww::Vector)
   @assert pcut > 0
   @assert N > 2

   # correct the transformation
   tin = transform(trans, rin)
   tcut = transform(trans, rcut)

   # normalise the weights
   ww = ww ./ sum(ww)
   # transform the rdf
   tdf = transform.(trans, rdf)
   dotw = (f1, f2) -> dot(f1, ww .* f2)
   dotxw = (f1, f2) -> dot(tdf .* f1, ww .* f2)

   # start the iteration
   J1 = (tdf .- tcut).^pcut
   # J2 = (t - B) J1
   B[2] = dotxw(J1, J1) / dotw(J1, J1)
   C[2] = 0
   J2 = (tdf .- B[2]) .* J1
   Jprev = J2
   Jpprev = J1

   for n = 3:N
      # Jn = (t - B) J_{n-1} - C J_{n-2}
      B[n] = dotxw(Jprev, Jprev) / dotw(Jprev, Jprev)
      C[n] = dotxw(Jprev, Jpprev) / dotw(Jpprev, Jpprev)
      Jprev, Jpprev = (tdf .- B[n]) .* Jprev - C[n] * Jpprev, Jprev
   end
end
