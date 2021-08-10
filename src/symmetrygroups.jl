


abstract type SymmetryGroup end 


struct NoSym <: SymmetryGroup 
end 


struct O3{LSYM, MSYM} <: SymmetryGroup
end

O3(lsym::Symbol = :l, msym::Symbol = :m) = O3{lsym, msym}()

lsym(G::O3{LSYM, MSYM}) where {LSYM, MSYM} = LSYM 

msym(G::O3{LSYM, MSYM}) where {LSYM, MSYM} = MSYM 


# struct O3O3
#    syms::SYMS
# end



is_refbasisfcn(G::O3, AA) = all( bi[msym(G)] == 0 for bi in AA )




function coupling_coeffs(symgrp::O3, bb, rotc::Rot3DCoeffs)
   # bb = [ b1, b2, b3, ... ]
   # bi = (μ = ..., n = ..., l = ..., m = ...)
   #    (μ, n) -> n; only the l and m are used in the angular basis
   if length(bb) == 0
      # return [1.0,], [bb,]
		error("correlation order 0 is currently not allowed")
   end
   # convert to a format that the Rotations3D implementation can understand
   # this utility function splits the bb = (b1, b2 ...) with each
   # b1 = (μ = ..., n = ..., l = ..., m = ...) into
   #    l, and a new n = (μ, n)
   ll, nn = _b2llnn(symgrp, bb)
   # now we can call the coupling coefficient construiction!!
   U, Ms = rpe_basis(rotc, nn, ll)

   # but now we need to convert the m spec back to complete basis function
   # specifications (provided by sending in a prototype b = bb[1])
   rpebs = [ _nnllmm2b(symgrp, bb[1], nn, ll, mm) for mm in Ms ]

   return U, rpebs
end


function rpe_basis(A::Rot3DCoeffs,
						 nn::SVector{N, TN},
						 ll::SVector{N, Int}) where {N, TN}
	Ure, Mre = Rotations3D.re_basis(A, ll)
	G = _gramian(nn, ll, Ure, Mre)
   S = svd(G)
   rk = rank(Diagonal(S.S); rtol =  1e-7)
	Urpe = S.U[:, 1:rk]'
	return Diagonal(sqrt.(S.S[1:rk])) * Urpe * Ure, Mre
end


function _gramian(nn, ll, Ure, Mre)
   N = length(nn)
   nre = size(Ure, 1)
   G = zeros(Complex{Float64}, nre, nre)
   for σ in permutations(1:N)
      if (nn[σ] != nn) || (ll[σ] != ll); continue; end
      for (iU1, mm1) in enumerate(Mre), (iU2, mm2) in enumerate(Mre)
         if mm1[σ] == mm2
            for i1 = 1:nre, i2 = 1:nre
               G[i1, i2] += coco_dot(Ure[i1, iU1], Ure[i2, iU2])
            end
         end
      end
   end
   return G
end



_nnllmm2b(G, b, nn, ll, mm) = [ _nlm2b(G, b, n, l, m) for (n, l, m) in zip(nn, ll, mm) ]

@generated function _nlm2b(G::O3{LSYM, MSYM}, b::NamedTuple{ALLKEYS}, 
                           n::NamedTuple{NKEYS}, 
                           l, m) where {LSYM, MSYM, ALLKEYS, NKEYS}
   code =
      ( "( _b = (" * prod("$(k) = n.$(k), " for k in NKEYS) 
               * "$(LSYM) = l, $(MSYM) = m ); "
         *
        " b = (" * prod("$(k) = _b.$(k), " for k in ALLKEYS) * ") )" )
   :( $(Meta.parse(code)) )
end


function _b2llnn(G::O3, bb)
   @assert all( iszero(b[msym(G)]) for b in bb )
   ll = SVector( [b[lsym(G)] for b in bb]... )
   nn = SVector( [_all_but_lm(G, b) for b in bb]... )
   return ll, nn
end

"""
return a NamedTuple containing all values in b except those corresponding
to l and m keys
"""
@generated function _all_but_lm(G::O3{LSYM, MSYM}, 
                                b::NamedTuple{NAMES}) where {LSYM, MSYM, NAMES}
   code = "n = ("
   for k in NAMES
      if !(k in (LSYM, MSYM))
         code *= "$(k) = b.$(k), "
      end
   end
   code *= ")"
   quote
      $(Meta.parse(code))
      n
   end
end
