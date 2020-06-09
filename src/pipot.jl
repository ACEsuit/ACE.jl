
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs.SphericalHarmonics: SHBasis, index_y
using StaticArrays
using JuLIP: AbstractCalculator, Atoms, JVec
using JuLIP.Potentials: SitePotential, SZList, ZList
using NeighbourLists: neigs

import JuLIP, JuLIP.MLIPs

export PIPotential


"""
`struct PIPotential` : specifies a PIPotential, which is basically defined
through a PIBasis and its coefficients
"""
struct PIPotential{T, NZ, TIN} <: SitePotential
   pibasis::TIN
   coeffs::NTuple{NZ, Vector{T}}
end

cutoff(V::PIPotential) = cutoff(V.pibasis)

==(V1::PIPotential, V2::PIPotential) =
      (V1.pibasis == V2.pibasis) && (V1.coeffs == V2.coeffs)

# TODO: this doesn't feel right ... should be real(T)?
Base.eltype(::PIPotential{T}) where {T} = real(T)

z2i(V::PIPotential, z::AtomicNumber) = z2i(V.pibasis, z)
JuLIP.numz(V::PIPotential) = numz(V.pibasis)





# ------------------------------------------------------------
#   Initialisation code
# ------------------------------------------------------------

function combine(basis::PIBasis, coeffs; evaluator = :classic)
   V = PIPotential(basis, coeffs)
   if evaluator == :classic
      return V
   elseif evaluator == :dag
      return GraphPIPot(V)
   end
   error("unknown evaluator $evaluator")
end


function PIPotential(basis::PIBasis, coeffs::Vector{<: Number})
   coeffs_t = ntuple(iz0 -> coeffs[basis.inner[iz0].AAindices], numz(basis))
   return PIPotential(basis, coeffs_t)
end


# ------------------------------------------------------------
#   FIO code
# ------------------------------------------------------------

write_dict(V::PIPotential) = Dict(
      "__id__" => "SHIPs_PIPotential",
     "pibasis" => write_dict(V.pibasis),
      "coeffs" => [ write_dict.(V.coeffs)... ] )
#    if ntests > 0
#       tests = SHIPs.Random.rand_nhd(Nat, J::ScalarBasis, species = :X)
#       Pr = V.pibasis.basis1p.J
#       for ntest = 1:5
#
#          r = SHIPs.rand_radial(V.pibasis.basis1p.J)
#          Pr = evaluate(B3.J, r)
#          push!(rtests, Dict("r" => r, "Pr" => Pr))
#       end
#
#    end
#    return D
# end

read_dict(::Val{:SHIPs_PIPotential}, D::Dict; tests = true) =
   PIPotential( read_dict(D["pibasis"]),
                tuple( read_dict.( D["coeffs"] )... ) )


# function test_compat(V::PIPotential, rtests, tests)
#
# end

# ------------------------------------------------------------
#   Evaluation code
# ------------------------------------------------------------


# TODO: generalise the R, Z, allocation
alloc_temp(V::PIPotential{T}, maxN::Integer) where {T} =
   (
      R = zeros(JVec{real(T)}, maxN),
      Z = zeros(AtomicNumber, maxN),
      tmp_pibasis = alloc_temp(V.pibasis, maxN),
  )



# compute one site energy
function evaluate!(tmp, V::PIPotential,
                   Rs::AbstractVector{JVec{T}},
                   Zs::AbstractVector{<:AtomicNumber},
                   z0::AtomicNumber) where {T}
   iz0 = z2i(V, z0)
   A = evaluate!(tmp.tmp_pibasis.A, tmp.tmp_pibasis.tmp_basis1p,
                 V.pibasis.basis1p, Rs, Zs, z0)
   inner = V.pibasis.inner[iz0]
   c = V.coeffs[iz0]
   Es = zero(T)
   @inbounds for iAA = 1:length(inner)
      Esi = one(Complex{T}) * c[iAA]    # TODO: OW - NASTY!!!
      for α = 1:inner.orders[iAA]
         Esi *= A[inner.iAA2iA[iAA, α]]
      end
      Es += real(Esi)
   end
   return Es
end

# TODO: generalise the R, Z, allocation
alloc_temp_d(V::PIPotential{T}, N::Integer) where {T} =
      (
      dAco = zeros(eltype(V.pibasis),
                   maximum(length(V.pibasis.basis1p, iz) for iz=1:numz(V))),
       tmpd_pibasis = alloc_temp_d(V.pibasis, N),
       dV = zeros(JVec{real(T)}, N),
        R = zeros(JVec{real(T)}, N),
        Z = zeros(AtomicNumber, N)
      )

# compute one site energy
function evaluate_d!(dEs, tmpd, V::PIPotential,
                     Rs::AbstractVector{<: JVec{T}},
                     Zs::AbstractVector{AtomicNumber},
                     z0::AtomicNumber
                     ) where {T}
   iz0 = z2i(V, z0)
   basis1p = V.pibasis.basis1p
   tmpd_1p = tmpd.tmpd_pibasis.tmpd_basis1p
   Araw = tmpd.tmpd_pibasis.A

   # stage 1: precompute all the A values
   A = evaluate!(Araw, tmpd_1p, basis1p, Rs, Zs, z0)

   # stage 2: compute the coefficients for the ∇A_{klm} = ∇ϕ_{klm}
   dAco = tmpd.dAco
   c = V.coeffs[iz0]
   inner = V.pibasis.inner[iz0]
   fill!(dAco, 0)
   for iAA = 1:length(inner)
      for α = 1:inner.orders[iAA]
         CxA_α = c[iAA]
         for β = 1:inner.orders[iAA]
            if β != α
               CxA_α *= A[inner.iAA2iA[iAA, β]]
            end
         end
         iAα = inner.iAA2iA[iAA, α]
         dAco[iAα] += CxA_α
      end
   end

   # stage 3: get the gradients
   fill!(dEs, zero(JVec{T}))
   dAraw = tmpd.tmpd_pibasis.dA
   for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
      evaluate_d!(Araw, dAraw, tmpd_1p, basis1p, R, Z, z0)
      iz = z2i(basis1p, Z)
      zinds = basis1p.Aindices[iz, iz0]
      for iA = 1:length(basis1p, iz, iz0)
         dEs[iR] += real(dAco[zinds[iA]] * dAraw[zinds[iA]])
      end
   end

   return dEs
end




# ------------------------------------------------------------
#    GraphPIPot -> temporary place for dag-evaluator wrapper

import SHIPs.DAG: CorrEvalGraph, get_eval_graph

graph_evaluator(pipot::PIPotential; kwargs...) =
      GraphPIPot(pipot; kwargs...)


struct GraphPIPot{T, TI, NZ, TB} <: SitePotential
   basis1p::TB
   dags::NTuple{NZ, CorrEvalGraph{T, TI}}
end

function GraphPIPot(pipot::PIPotential; kwargs...)
   dags = [ get_eval_graph(pipot.pibasis.inner[iz], pipot.coeffs[iz];
                           kwargs...)  for iz = 1:numz(pipot) ]
   return GraphPIPot(pipot.pibasis.basis1p, tuple(dags...))
end


i2z(V::GraphPIPot, i::Integer) = i2z(V.basis1p, i)
z2i(V::GraphPIPot, z::AtomicNumber) = z2i(V.basis1p, z)
numz(V::GraphPIPot) = numz(V.basis1p)

cutoff(V::GraphPIPot) = cutoff(V.basis1p)

Base.eltype(V::GraphPIPot{T}) where {T} = real(T)

_maxstore(V::GraphPIPot) = maximum( dag.numstore for dag in V.dags )

alloc_temp(V::GraphPIPot{T}, maxN::Integer) where {T} =
   (
   R = zeros(JVec{real(T)}, maxN),
   Z = zeros(AtomicNumber, maxN),
   tmp_basis1p = alloc_temp(V.basis1p),
   AA = zeros(eltype(V.basis1p), _maxstore(V)),
   A = alloc_B(V.basis1p)
    )


# function evaluate!(tmp, V::GraphPIPot, Rs, Zs, z0)
#    AAdag = tmp.AA
#    A = tmp.A
#    iz0 = z2i(V, z0)
#    evaluate!(A, tmp.tmp_basis1p, V.basis1p, Rs, Zs, z0)
#    Es = zero(eltype(V))
#    traverse_fwd!(AAdag, V.dags[iz0], A,
#                  (coeff, AAval) -> (Es = muladd(coeff, real(AAval), Es)))
#    return Es
# end


function evaluate!(tmp, V::GraphPIPot, Rs, Zs, z0)
   AAdag = tmp.AA
   A = tmp.A
   iz0 = z2i(V, z0)
   dag = V.dags[iz0]
   nodes = dag.nodes
   vals = dag.vals
   @assert length(A) >= dag.num1
   @assert length(AAdag) >= dag.numstore

   evaluate!(A, tmp.tmp_basis1p, V.basis1p, Rs, Zs, z0)

   Es = zero(eltype(V))
   @inbounds for i = 1:dag.num1
      AAdag[i] = a = A[i]
      Es = muladd(vals[i], real(a), Es)
   end

   @inbounds for i = (dag.num1+1):dag.numstore
      n1, n2 = nodes[i]
      AAdag[i] = a = AAdag[n1] * AAdag[n2]
      Es = muladd(vals[i], real(a), Es)
   end

   @inbounds for i = (dag.numstore+1):length(dag)
      n1, n2 = nodes[i]
      a = AAdag[n1] * AAdag[n2]
      Es = muladd(vals[i], real(a), Es)
   end

   return Es
end



alloc_temp_d(V::GraphPIPot{T}, maxN::Integer) where {T} =
   (
   R = zeros(JVec{real(T)}, maxN),
   Z = zeros(AtomicNumber, maxN),
    dV = zeros(JVec{real(T)}, maxN),
   tmpd_basis1p = alloc_temp_d(V.basis1p),
   AA = zeros(eltype(V.basis1p), _maxstore(V)),
   B = zeros(eltype(V.basis1p), _maxstore(V)),
   A = alloc_B(V.basis1p),
   dA = alloc_dB(V.basis1p)
    )


function evaluate_d!(dEs, tmpd, V::GraphPIPot{T}, Rs, Zs, z0) where {T}
   iz0 = z2i(V, z0)
   AA = tmpd.AA
   B = tmpd.B
   tmpd_basis1p = tmpd.tmpd_basis1p
   basis1p = V.basis1p
   dag = V.dags[iz0]
   nodes = dag.nodes
   coeffs = dag.vals

   # we start from the representation
   #     V = sum B[i] AA[i]
   # i.e. this vector represents the constributions c[i] ∂AA[i]
   copy!(B, coeffs)

   # FORWARD PASS
   # ------------
   # evaluate the 1-particle basis
   # this puts the first `dag.num1` 1-b (trivial) correlations into the
   # storage array, and from these we can build the rest
   evaluate!(AA, tmpd_basis1p, basis1p, Rs, Zs, z0)

   # Stage 2 of evaluate!
   # go through the dag and store the intermediate results we need
   @inbounds @fastmath for i = (dag.num1+1):dag.numstore
      n1, n2 = nodes[i]
      AA[i] = muladd(AA[n1], AA[n2], AA[i])
   end

   # BACKWARD PASS
   # --------------
   # fill the B array -> coefficients of the derivatives
   #  AA_i = AA_{n1} * AA_{n2}
   #  ∂AA_i = AA_{n1} * ∂AA_{n2} + AA_{n1} * AA_{n2}
   #  c_{n1} * ∂AA_{n1} <- (c_{n1} + c_i AA_{n2}) ∂AA_{n1}
   @inbounds @fastmath for i = length(dag):-1:(dag.numstore+1)
      c = coeffs[i]
      n1, n2 = nodes[i]
      B[n1] = muladd(c, AA[n2], B[n1])
      B[n2] = muladd(c, AA[n1], B[n2])
   end
   # in stage 2 c = C[i] is replaced with b = B[i]
   @inbounds @fastmath for i = dag.numstore:-1:(dag.num1+1)
      n1, n2 = nodes[i]
      b = B[i]
      B[n1] = muladd(b, AA[n2], B[n1])
      B[n2] = muladd(b, AA[n1], B[n2])
   end

   # stage 3: get the gradients
   fill!(dEs, zero(JVec{T}))
   A = tmpd.A
   dA = tmpd.dA
   for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
      evaluate_d!(A, dA, tmpd_basis1p, basis1p, R, Z, z0)
      iz = z2i(basis1p, Z)
      inds = basis1p.Aindices[iz, iz0]
      for iA = 1:length(basis1p, iz, iz0)
         dEs[iR] += real(B[inds[iA]] * dA[inds[iA]])
      end
   end

   return dEs
end
