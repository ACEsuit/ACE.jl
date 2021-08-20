


# NOTE: at the moment these are ad hoc implementations for each group that 
#       we decide we need. eventually we can hopefully simplify and merge 
#       many of these codes
#       The first version was written before NamedTupleTools (or before I 
#        knew about it -> this will simplify the code here a bit)


using NamedTupleTools: delete, merge, namedtuple

abstract type SymmetryGroup end 


"""
`struct NoSym <: SymmetryGroup ` : no symmetrisation other than 
permutation symmetry already baked into the AA basis.
"""

"""
`struct NoSym <: SymmetryGroup` : no symmetry beyond the standard 
permutation symmetry. This is currently not used, but could be incorporated 
to provide a more streamlined experience for the user. 
"""
struct NoSym <: SymmetryGroup 
end 

# -----------------------  O3 SYMMETRY 

# this is a prototype implemenation; eventually (asap!) we need to allow 
# rotation of multiple features at once, e.g., spin-orbit coupling!

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


write_dict(G::O3) = 
      Dict("__id__" => "ACE_O3", 
           "lsym" => lsym(G), 
           "msym" => msym(G) )

read_dict(::Val{:ACE_O3}, D::Dict) = 
      O3(Symbol(D["lsym"]), Symbol(D["msym"]))


is_refbasisfcn(G::O3, AA) = all( bi[msym(G)] == 0 for bi in AA )

get_sym_spec(G::O3, bb) = delete.(bb, (msym(G),))


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

# this is a preliminary implementation; eventually we may want a more 
# general description composition of arbitrary isometry combinations 

"""
`struct O3O3 <: SymmetryGroup` : This type implements the ``O(3,3)`` symmetry 
group. This is useful when a particle has two euclidean vector attributes, say 
``{\\bm r}`` and ``{\\bm s}`` and the action of the group on the pair is 
```math 
   (Q_r, Q_s)[ (\\boldsymbol{r}, \\boldsymbol{s}) ] = (Q_r \\boldsymbol{r}, Q_s \\boldsymbol{s})
```
A canA canonical application is magnetism: it is known that spin-orbit coupling 
is a very weak effect. By ignoring it, i.e., letting positions and spins rotate 
independently of one another, one makes a small modelling error. This leads 
precisely to the ``O(3,3)`` symmetry. 

To construct this group, use 
```julia
symgrp = O3(:lr, :mr) ⊗ O3(:ls, :ms)
```
or replace those symbols with the appropriate symbols used to specify the 
corresponding `Ylm1pbasis` objects. 
"""
struct O3O3{LSYM1, MSYM1, LSYM2, MSYM2} <: SymmetryGroup
   G1::O3{LSYM1, MSYM1} 
   G2::O3{LSYM2, MSYM2}
end

import Base: kron

function kron(G1::O3, G2::O3)
   @assert lsym(G1) != lsym(G2) 
   @assert msym(G1) != msym(G2)
   return O3O3(G1, G2)
end

⊗(G1::O3, G2::O3) = kron(G1, G2)
export ⊗

write_dict(G::O3O3) = 
      Dict("__id__" => "ACE_O3O3", 
           "G1" => write_dict(G.G1), 
           "G2" => write_dict(G.G2) )

read_dict(::Val{:ACE_O3O3}, D::Dict) = 
      read_dict(D["G1"]) ⊗ read_dict(D["G2"])



is_refbasisfcn(G::O3O3, AA) = all( bi[msym(grp)] == 0 
                                   for bi in AA, grp in (G.G1, G.G2) )

get_sym_spec(G::O3O3, bb) = delete.(bb, Ref( (msym(G.G1), msym(G.G2)) ))

function coupling_coeffs(symgrp::O3O3, bb, rotc::Rot3DCoeffs)
   # bb = [ b1, b2, b3, ... ]
   # bi = (μ = ..., n = ..., l1 = ..., m1 = ..., l2 = ..., m2 = ...)
   #    (μ, n, ...) -> n; only the l and m are used in the angular basis
   if length(bb) == 0
      error("correlation order 0 is currently not allowed")
   end

   # the prototype namedtuple describing a single 1p basis fcn 
   PROTOTUPLE = prototype(bb[1])
   NU = length(bb)

   # convert to (nn, ll, mm) format for Rotations3D
   ll1, ll2, nn, ll12 = _b2llnn(symgrp, bb)
   # ... and construct the coupling coefficients for the individual subgroups 
   U1, M1 = Rotations3D.re_basis(rotc, ll1)
   U2, M2 = Rotations3D.re_basis(rotc, ll2)

   nU1, nM1 = size(U1)
   nU2, nM2 = size(U2)
   @assert nM1 == length(M1) 
   @assert nM2 == length(M2)
   UT = promote_type(eltype(U1), eltype(U2))

   # there is admissible combination: 
   if nU1 == 0 || nU2 == 0  
      return UT[], SVector{NU, PROTOTUPLE}[]
   end

   # now combine them into the effective coupling coeffs and combined Ms 
   # each column Ure[:, i] corresponds to one rotation-invariant basis fcn 
   M1M2TUPLE = namedtuple(msym(symgrp.G1), msym(symgrp.G2))
   Mre = Vector{typeof(Vector(M1M2TUPLE.(M1[1], M2[1])))}(undef, nM1 * nM2)
   Ure = zeros( UT, (nU1 * nU2, nM1 * nM2) )
   jdx = 0
   for j1 = 1:nM1, j2 = 1:nM2
      jdx += 1
      Mre[jdx] = Vector(M1M2TUPLE.(M1[j1], M2[j2]))
      idx = 0 
      for i1 = 1:nU1, i2 = 1:nU2
         idx += 1
         Ure[idx, jdx] = U1[i1, j1] * U2[i2, j2]
      end
   end

   # insert another reduction step 
   # in my tests this never reduces the size, but I haven't had the time
   # to actually prove it isn't needed, so we will keep it for now 
   Gre = [ sum(coco_dot.(Ure[i1, :], Ure[i2, :])) for i1 = 1:size(Ure, 1), i2 = 1:size(Ure, 1) ]
   Sre = svd(Gre)
   rk = rank(Diagonal(Sre.S); rtol =  1e-7)
   Ure = Sre.U[:, 1:rk]' * Ure 

   # now symmetrize w.r.t. permutations 
   G = _gramian(nn, ll12, Ure, Mre)
   S = svd(G)
   rk = rank(Diagonal(S.S); rtol =  1e-7)
   Urpe = S.U[:, 1:rk]'
   U = Diagonal(sqrt.(S.S[1:rk])) * Urpe * Ure
 
   # reconstruct the basis function specifications
   rpebs = [ PROTOTUPLE.(merge.(nn, ll12, mm12)) for mm12 in Mre ]
   
   return U, rpebs
end

function _b2llnn(G::O3O3{L1, M1, L2, M2}, bb) where {L1, M1, L2, M2}
   N = length(bb)
   @assert all( iszero(b[M]) for b in bb, M in (M1, M2) )
   ll1 = ntuple(i -> bb[i][L1], N)  |> SVector
   ll2 = ntuple(i -> bb[i][L2], N)  |> SVector
   nn  = ntuple(i -> delete(bb[i], (L1, M1, L2, M2)), N)
   ll12 = ntuple(i -> select(bb[i], (L1, L2)), N)
   return ll1, ll2, nn, ll12 
end

