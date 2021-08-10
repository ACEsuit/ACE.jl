


# NOTE: at the moment these are ad hoc implementations for each group that 
#       we decide we need. eventually we can hopefully simplify and merge 
#       many of these codes


using NamedTupleTools: delete, merge, namedtuple

abstract type SymmetryGroup end 


"""
`struct NoSym <: SymmetryGroup ` : no symmetrisation other than 
permutation symmetry already baked into the AA basis.
"""

struct NoSym <: SymmetryGroup 
end 

# -----------------------  O3 SYMMETRY 
"""
`struct O3 <: SymmetryGroup` : this is the default symmetry group; describing 
the action of a single O3 group on the basis. 

Standard Usage: 
```julia 
O3()
```
will create an `O3{:l, :m}` instance, i.e. the group will expect the symbols 
`:l, :m` in the relevant 1p basis. 

But if the Ylm component of the 1p basis uses different symbols then one can 
tell `O3` this via `O3(lsym, rsym)`. E.g. if the variable w.r.t. which we 
symmetrize is a spin `s` then we might call it `O3(:ls, :ms)`. The main thing 
to remember is that the symbols in the Ylm basis and in the O3 basis must 
match. 
"""
struct O3{LSYM, MSYM} <: SymmetryGroup
end

O3(lsym::Symbol = :l, msym::Symbol = :m) = O3{lsym, msym}()

lsym(G::O3{LSYM, MSYM}) where {LSYM, MSYM} = LSYM 
getl(G::O3, b::NamedTuple) = b[lsym(G)]

msym(G::O3{LSYM, MSYM}) where {LSYM, MSYM} = MSYM 
getm(G::O3, b::NamedTuple) = b[msym(G)]


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


# TODO: replace all this awful code with NamedTupleTools

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

TODO: get rid of this ridiculousness and replace with NamedTupleTools methods 
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


# -------------- O3 ⊗ O3 

struct O3O3{LSYM1, MSYM1, LSYM2, MSYM2} 
   G1::O3{LSYM1, MSYM1} 
   G2::O3{LSYM2, MSYM2}
end

import Base: ⊗ 
function ⊗(G1::O3, G2::O3)
   @assert lsym(G1) != lsym(G2) 
   @assert msym(G1) != msym(G2)
   return O3O3(G1, G2)
end


is_refbasisfcn(G::O3O3, AA) = all( bi[msym(grp)] == 0 
                                   for bi in AA, grp in (G.G1, G.G2) )


function coupling_coeffs(symgrp::O3O3, bb, rotc::Rot3DCoeffs)
   # bb = [ b1, b2, b3, ... ]
   # bi = (μ = ..., n = ..., l1 = ..., m1 = ..., l2 = ..., m2 = ...)
   #    (μ, n, ...) -> n; only the l and m are used in the angular basis
   if length(bb) == 0
      error("correlation order 0 is currently not allowed")
   end

   # convert to (nn, ll, mm) format for Rotations3D
   ll1, ll2, nn = _b2llnn(symgrp, bb)
   # ... and construct the coupling coefficients for the individual subgroups 
   U1, M1 = Rotations3D.re_basis(rotc, ll1)
   U2, M2 = Rotations3D.re_basis(rotc, ll2)

   # now combine them into the effective coupling coeffs 
   Ure = [ U1[i1] * U2[i2] for i1 = 1:length(U1), i2 = 1:length(U2) ]
   Mre = [ _nnllmm2b(G, nn, ll1, M1[i1], ll2, M2[i2])
           for i1 = 1:length(U1), i2 = 1:length(U2) ]

   # now symmetrize w.r.t. permutations 

   # but now we need to convert the m spec back to complete basis function
   # specifications (provided by sending in a prototype b = bb[1])
   rpebs = [ _nnllmm2b(symgrp, bb[1], nn, ll, mm) for mm in Ms ]

   return U, rpebs
end

function _b2llnn(G::O3O3{L1, M1, L2, M2}, bb) where {L1, M1, L2, M2}
   @assert all( iszero(b[M]) for b in bb, M in (M1, M2) )
   ll1 = ntuple( i -> bb[i][L1], length(bb) )  |> SVector
   ll2 = ntuple( i -> bb[i][L2], length(bb) )  |> SVector
   nn = ntuple( i -> delete(bb[i], (L1, M1, L2, M2)), length(bb) )
   return ll1, ll2, nn 
end

function _nnllmm2b(G::O3O3{L1, M1, L2, M2}, nn, ll1, mm1, ll2, mm2
                  ) where {L1, M1, L2, M2}
   NTPROTO = namedtuple(L1, M1, L2, M2)
   return ntuple( i -> merge(nn[i], NTPROTO(ll1[i], mm1[i], ll2[i], mm2[i])), 
                  length(nn) )
end