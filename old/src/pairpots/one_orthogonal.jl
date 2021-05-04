
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using LinearAlgebra: norm, dot

import JuLIP: evaluate!, evaluate_d!, alloc_temp
import JuLIP.MLIPs: alloc_B, alloc_dB, IPBasis

struct OneOrthogonal{T,B} <: IPBasis
   coeffs::Matrix{T}
   nested::B
   # ----------------- used only for construction ...
   #                   but useful to have since it defines the notion or orth.
   tdf::Vector{T}
   ww::Vector{T}
end


Base.length(P::OneOrthogonal) = length(P.nested)

ACE.rand_radial(J::OneOrthogonal) = ACE.rand_radial(J.nested)


"""
    OneOrthogonal(P, tdf = P.tdf, ww = P.ww)

Orthogonal basis `Q` such that `span(Q) == span(P)` and `Q[:,1:end-1]` are
orthogonal to `1`, where `P` is an orthogonal basis with respect to the inner
product defined by `(tdf,ww)`.
"""
function OneOrthogonal(P, tdf=P.tdf, ww = P.ww)
   T = eltype(alloc_B(P))
   dotw = (f1, f2) -> dot(f1,ww.*f2)
   normw = f->sqrt(dotw(f1,f2))

   tmp = alloc_temp(P)
   p = Matrix{T}(undef, length(tdf), length(P))
   for i = 1:length(tdf)
      evaluate!(@view(p[i,:]), tmp, P, tdf[i])
   end

   rotate! = (qe,p1,p2) -> begin
      local c = (dot(p1,ww), dot(p2,ww))
      c = c ./ norm(c)
      qe .= c[1]*p1 .+ c[2]*p2
      return c
   end

   qe = Vector{T}(undef, length(tdf))
   c = Matrix{T}(undef, 2, length(P)-1)
   c[:,1] .= rotate!(qe, p[:,1], p[:,2])
   for j = 2:length(P)-1
      c[:,j] .= rotate!(qe, p[:,j+1],qe)
   end

   return OneOrthogonal(c, P, tdf, ww)
end

alloc_B( J::OneOrthogonal{T}) where {T} = zeros(T, length(J))
alloc_dB(J::OneOrthogonal{T}) where {T} = zeros(T, length(J))

alloc_B( J::OneOrthogonal{T}, x::TX) where {T, TX} = zeros(TX, length(J))
alloc_dB(J::OneOrthogonal{T}, x::TX) where {T, TX} = zeros(TX, length(J))

function evaluate!(q, tmp, Q::OneOrthogonal, t)
   P = Q.nested
   c = Q.coeffs
   evaluate!(q, tmp, P, t)
   q[1],qe = c[2,1]*q[1] - c[1,1]*q[2], c[1,1]*q[1] + c[2,1]*q[2]
   for j = 2:length(Q)-1
       q[j],qe = c[2,j]*q[j+1] - c[1,j]*qe, c[1,j]*q[j+1] + c[2,j]*qe
   end
   q[end] = qe
   return q
end

function evaluate_d!(q, dq, tmp, Q::OneOrthogonal, t)
   P = Q.nested
   c = Q.coeffs
   evaluate_d!(q, dq, tmp, P, t)
    q[1], qe = c[2,1]* q[1] - c[1,1]* q[2], c[1,1]* q[1] + c[2,1]* q[2]
   dq[1],dqe = c[2,1]*dq[1] - c[1,1]*dq[2], c[1,1]*dq[1] + c[2,1]*dq[2]
   for j = 2:length(Q)-1
        q[j], qe = c[2,j]* q[j+1] - c[1,j]* qe, c[1,j]* q[j+1] + c[2,j]* qe
       dq[j],dqe = c[2,j]*dq[j+1] - c[1,j]*dqe, c[1,j]*dq[j+1] + c[2,j]*dqe
   end
    q[end] = qe
   dq[end] = dqe
   return dq
end


# ------------------------------------------------------------------
#   a filtering mechanism to sort out the non-orthogonal functions

function filter_oneorth(ν, P::OneOrthogonal)
   degP = length(P)-1
   # accept all ν with ki < degP, i.e., don't allow the last function
   return maximum(ν.kk) < degP
end
