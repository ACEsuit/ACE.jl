
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



module Export

import SHIPs
using SHIPs.RPI: BasicPSH1pBasis, PSH1pBasisFcn
using SHIPs: PIBasis, PIBasisFcn, PIPotential
using SHIPs.OrthPolys: TransformedPolys
using SHIPs: rand_radial, cutoff, numz

function export_ace(fname::AbstractString, V;  kwargs...)
   fptr = open(fname; write=true)
   export_ace(fptr, V; kwargs...)
   close(fptr)
end


function export_ace(fptr::IOStream, Pr::TransformedPolys; ntests=0, kwargs...)
   p = Pr.trans.p
   r0 = Pr.trans.r0
   xr = Pr.J.tr
   xl = Pr.J.tl
   pr = Pr.J.pr
   pl = Pr.J.pl
   rcut = cutoff(Pr)
   maxn = length(Pr)

   println(fptr, "radial basis: id=ships")
   println(fptr, "transform parameters: p=$(p) r0=$(r0)")
   println(fptr, "cutoff parameters: rcut=$rcut xl=$xl xr=$xr pl=$pl pr=$pr")
   println(fptr, "recursion coefficients: maxn=$(maxn)")
   for n = 1:maxn
      println(fptr, " $(Pr.J.A[n]) $(Pr.J.B[n]) $(Pr.J.C[n])")
   end

   # save some tests
   println(fptr, "tests: ntests=$(ntests)")
   for itest = 1:ntests
      r = SHIPs.rand_radial(Pr)
      P = SHIPs.evaluate(Pr, r)
      println(fptr, " r=$(r)")
      for n = 1:length(P)
         println(fptr, " $(P[n])")
      end
   end
end


function export_ace(fptr::IOStream, V::PIPotential; kwargs...)
   @assert numz(V) == 1
   inner = V.pibasis.inner[1]
   coeffs = V.coeffs[1]
   # sort the basis functions into groups the way the ace evaluator wants it
   groups = _basis_groups(inner, coeffs)

   # export_ace the radial basis
   export_ace(fptr, V.pibasis.basis1p.J; kwargs...)

   # an empty line to separate it from the n-body basis
   println(fptr, "")

   # header
   println(fptr, "num_c_tilde_max=$(length(groups))")
   num_ms_combinations_max = maximum( length(g["M"]) for g in groups )
   println(fptr, "num_ms_combinations_max=$(num_ms_combinations_max)")

   # write the pair basis groups
   total_basis_size_rank1 = sum( length(g["l"] ==  1) for g in groups )
   println(fptr, "total_basis_size_rank1: $(total_basis_size_rank1)")
   for i = 1:total_basis_size_rank1
      g = groups[i]
      _write_group(fptr, g)
   end

   # write the rest
   total_basis_size = length(groups) - total_basis_size_rank1
   println(fptr, "total_basis_size: $(total_basis_size)")
   for i = (total_basis_size_rank1+1):length(groups)
      g = groups[i]
      _write_group(fptr, g)
   end

end

function _write_group(fptr, g)
   order = length(g["l"])
   println(fptr, "ctilde_basis_func: rank=$(order) ndens=1 mu0=0 mu=(" * " 0 "^order * ")")
   println(fptr, "n=(" * prod(" $(ni) " for ni in g["n"]) * ")")
   println(fptr, "l=(" * prod(" $(li) " for li in g["l"]) * ")")
   println(fptr, "num_ms=$(length(g["M"]))")
   for (m, c) in zip(g["M"], g["C"])
      println(fptr, "<" * prod(" $(mi) " for mi in m) * ">:  $(c)")
   end
end


#
# function export_tests(fptr::IOStream, basis::PIBasis)
#
# end



function _basis_groups(inner, coeffs)
   allspec = collect(keys(inner.b2iAA))
   NL = [ ( [b1.n for b1 in b.oneps], [b1.l for b1 in b.oneps] ) for b in allspec ]
   M = [ [b1.m for b1 in b.oneps] for b in allspec ]
   ords = length.(M)
   perm = sortperm(ords)
   NL = NL[perm]
   M = M[perm]
   C = coeffs[perm]
   @assert issorted(length.(M))
   bgrps = []
   alldone = fill(false, length(NL))
   for i = 1:length(NL)
      if alldone[i]; continue; end
      nl = NL[i]
      Inl = findall(NL .== Ref(nl))
      Mnl = M[Inl]
      Cnl = C[Inl]
      pnl = sortperm(Mnl)
      Mnl = Mnl[pnl]
      Cnl = Cnl[pnl]
      push!(bgrps, Dict("n" => nl[1], "l" => nl[2],
                        "M" => Mnl, "C" => Cnl))
   end
   return bgrps
end


end
