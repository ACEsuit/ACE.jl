
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


module Import


using ACE, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d
using ACE: PIBasisFcn, DAGEvaluator
using ACE.RPI: BasicPSH1pBasis, PSH1pBasisFcn


function import_rbasis_v05(D, rtests = [])
   @assert D["__id__"] == "SHIPs_TransformedJacobi"
   trans = ACE.Transforms.PolyTransform(D["trans"])
   ru, rl = D["ru"], D["rl"]
   tu, tl = trans(ru), trans(rl)
   cutoff_id = D["cutoff"]["__id__"]
   if cutoff_id == "SHIPs_PolyCutoff1s"
      pl = 0
      pu = D["cutoff"]["P"]
   elseif cutoff_id == "SHIPs_PolyCutoff2s"
      pl = pu = D["cutoff"]["P"]
   else
      error("unknown cutoff %(cutoff_id)")
   end
   if tl > tu
      tl, tr = tu, tl
      pl, pr = pu, pl
   else
      tl, tr = tl, tu
      pl, pr = pl, pu
   end

   A = Float64.(D["jacobicoeffs"]["A"])
   B = Float64.(D["jacobicoeffs"]["B"])
   C = Float64.(D["jacobicoeffs"]["C"])
   nrm = Float64.(D["jacobicoeffs"]["nrm"])
   deg = D["deg"]
   α, β = D["a"], D["b"]

   An = zeros(length(A) + 1)
   Bn = zeros(length(An))
   Cn = zeros(length(An))
   An[1] = nrm[1]
   An[2] = -nrm[2] / nrm[1] * 0.5 * (α + β + 2)
   Bn[2] = nrm[2] / nrm[1] * ((α + 1) - 0.5 * (α + β + 2))
   for n = 3:length(An)
      An[n] = -nrm[n] / nrm[n-1] * A[n-1]
      Bn[n] = nrm[n] / nrm[n-1] * B[n-1]
      Cn[n] = nrm[n] / nrm[n-2] * C[n-1]
   end

   # transform to the correct interval
   a = (2 / (tr - tl))
   At = An * a
   Bt = Bn - An * ((tr + tl) / (tr - tl))
   Ct = Cn
   At[1] = An[1] * a^(pl+pr)

   Jt = ACE.OrthPolys.OrthPolyBasis(pl, tl, pr, tr, At, Bt, Ct,
                                      Float64[], Float64[])
   Jr = ACE.OrthPolys.TransformedPolys(Jt, trans, rl, ru)

   if !isempty(rtests) > 0
      r = rtests[1]["r"]
      Jr_test = rtests[1]["Pr"]
      Jr_new = evaluate(Jr, r)
      f = Jr_test[1] / Jr_new[1]
      Jr.J.A[1] *= f
   end
   return Jr
end

import_pipot_v05(fname::AbstractString) =
      import_pipot_v05(JuLIP.read_dict(fname))

function import_pipot_v05(D::Dict)
   species = D["zlist"]["list"]
   zlist = read_dict(D["zlist"])
   rawcoeffs = Float64.(D["coeffs_re"][1])

   # get the radial basis
   Pr = import_rbasis_v05(D["J"], D["rtests"])

   # get the 1-p and n-p basis specifications
   rawspec = D["aalists"][1]["ZKLM_list"]

   pispec = []
   pispec_i = []
   pispec_i_dict = Dict{Vector{Int}, Int}()
   spec1 = []
   coeffs = []
   for (idx, ZNLM) in enumerate(rawspec)
      izs = ZNLM[1]
      ns = ZNLM[2] .+ 1
      ls = ZNLM[3]
      ms = ZNLM[4]
      order = length(ns)
      zs = [AtomicNumber(species[iz]) for iz in izs]
      bs = PSH1pBasisFcn[]
      bs_i = Int[]
      for α = 1:order
         b = PSH1pBasisFcn(ns[α], ls[α], ms[α], zs[α])
         I1 = findall(isequal(b), spec1)
         if isempty(I1)
            push!(spec1, b)
            i1 = length(spec1)
         else
            @assert length(I1) == 1
            i1 = I1[1]
         end
         push!(bs, b)
         push!(bs_i, i1)
      end
      p = sortperm(bs_i)
      bs_i = bs_i[p]
      bb = PIBasisFcn(AtomicNumber(species[1]), bs[p])

      if haskey(pispec_i_dict, bs_i)
         In = pispec_i_dict[bs_i]
         coeffs[In] += rawcoeffs[idx]
      else
         push!(pispec, bb)
         push!(pispec_i, bs_i)
         push!(coeffs, rawcoeffs[idx])
         pispec_i_dict[bs_i] = length(pispec_i)
      end

   end

   # construct 1p basis from the specs
   basis1p = BasicPSH1pBasis(Pr, zlist, identity.(spec1))
   # build a naive index allocation (we have only one species!!)
   AAindices = 1:length(pispec)
   # construct the inner PI basis from the specification
   inner = ACE.InnerPIBasis(spec1, pispec, AAindices, zlist.list[1])
   # ... and finally put it all together
   pibasis = PIBasis(basis1p, zlist, (inner,), DAGEvaluator())

   # now the inner basis has sorted the basis functins so we need to
   # fix the permutation of the coefficients
   perm = [ inner.b2iAA[b]  for b in pispec ]
   invperm = sortperm(perm)
   V = PIPotential(pibasis, (Float64.(coeffs[invperm]),))

   return V
end


end
