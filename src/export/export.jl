

module Export

import SHIPs, JuLIP
using SHIPs.RPI: BasicPSH1pBasis, PSH1pBasisFcn
using SHIPs: PIBasis, PIBasisFcn, PIPotential
using SHIPs.OrthPolys: TransformedPolys
using SHIPs: rand_radial, cutoff, numz
using JuLIP: energy, bulk, i2z, z2i, chemical_symbol

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

   println(fptr, "radbasename=SHIPsBasic")
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
      dP = SHIPs.evaluate_d(Pr, r)
      println(fptr, " r=$(r)")
      for n = 1:length(P)
         println(fptr, " $(P[n]) $(dP[n])")
      end
   end
end


function export_ace(fptr::IOStream, V::PIPotential; kwargs...)
   @assert numz(V) == 1
   inner = V.pibasis.inner[1]
   coeffs = V.coeffs[1]
   # sort the basis functions into groups the way the ace evaluator wants it
   groups = _basis_groups(inner, coeffs)
   lmax = maximum(maximum(g["l"]) for g in groups)

   # header
   println(fptr, "nelements=1")
   println(fptr, "elements: $(chemical_symbol(i2z(V, 1)))")
   println(fptr, "")
   println(fptr, "lmax=$lmax")
   println(fptr, "")
   println(fptr, "2 FS parameters:  1.000000 1.000000")
   println(fptr, "core energy-cutoff parameters: 100000.000000000000000000 1.000000000000000000")
   println(fptr, "")

   # export_ace the radial basis
   export_ace(fptr, V.pibasis.basis1p.J; kwargs...)

   # an empty line to separate it from the n-body basis
   println(fptr, "")

   # header
   rankmax = maximum(length(g["l"]) for g in groups)
   println(fptr, "rankmax=$rankmax")
   println(fptr, "ndensitymax=1")
   println(fptr, "")

   # header for basis list pair contributions
   println(fptr, "num_c_tilde_max=$(length(groups))")
   num_ms_combinations_max = maximum( length(g["M"]) for g in groups )
   println(fptr, "num_ms_combinations_max=$(num_ms_combinations_max)")

   # write the pair basis groups
   total_basis_size_rank1 = sum( (length(g["l"]) ==  1) for g in groups )
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
      c_ace = c / (4*π)^(order/2)
      println(fptr, "<" * prod(" $(mi) " for mi in m) * ">:  $(c_ace)")
   end
end



function export_ace_tests(fname::AbstractString, V::PIPotential, ntests = 1;
                          nrepeat = 3)
   s = JuLIP.chemical_symbol(i2z(V, 1))
   at = bulk(s, cubic=true, pbc=false) * nrepeat
   r0 = JuLIP.rnn(s)
   for n = 1:ntests
      JuLIP.rattle!(at, 0.05 * r0)
      E = energy(V, at)
      _write_test(fname * "_$n.dat", JuLIP.positions(at), E)
   end
end

function _write_test(fname, X, E)
   fptr = open(fname; write=true)
   println(fptr, "E = $E")
   println(fptr, "natoms = $(length(X))")
   println(fptr, "# type x y z")
   for n = 1:length(X)
      println(fptr, "0 $(X[n][1]) $(X[n][2]) $(X[n][2])")
   end
   close(fptr)
end



function _basis_groups(inner, coeffs)
   NL = []
   M = []
   C = []
   for b in keys(inner.b2iAA)
      if coeffs[ inner.b2iAA[b] ] != 0
         push!(NL, ( [b1.n for b1 in b.oneps], [b1.l for b1 in b.oneps] ))
         push!(M, [b1.m for b1 in b.oneps])
         push!(C, coeffs[ inner.b2iAA[b] ])
      end
   end
   ords = length.(M)
   perm = sortperm(ords)
   NL = NL[perm]
   M = M[perm]
   C = C[perm]
   @assert issorted(length.(M))
   bgrps = []
   alldone = fill(false, length(NL))
   for i = 1:length(NL)
      if alldone[i]; continue; end
      nl = NL[i]
      Inl = findall(NL .== Ref(nl))
      alldone[Inl] .= true
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
