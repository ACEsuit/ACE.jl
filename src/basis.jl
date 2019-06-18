
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------

using JuLIP, StaticArrays, LinearAlgebra


# TODO: Idea => replace (deg, wY) by a "Degree" type and dispatch a lot of
#       functionality on that => e.g. we can then try hyperbolic cross, etc.

using SHIPs.SphericalHarmonics: SHBasis, sizeY, SVec3, cart2spher, index_y,
         ClebschGordan

export SHIPBasis


# -------------------------------------------------------------
#       construct l,k tuples that specify basis functions
# -------------------------------------------------------------

function generate_LK(deg, wY::Real)
   allKL = NamedTuple{(:k, :l, :deg),Tuple{Int,Int,Float64}}[]
   degs = Float64[]
   # k + wY * l <= deg
   for k = 0:deg, l = 0:floor(Int, (deg-k)/wY)
      push!(allKL, (k=k, l=l, deg=(k+wY*l)))
      push!(degs, (k+wY*l))
   end
   # sort allKL according to total degree
   I = sortperm(degs)
   return allKL[I], degs[I]
end

function filter_tuples(KL, Nu, ::Val{2})  # 3B version
   keep = fill(true, length(Nu))
   for (i, ν) in enumerate(Nu)
      kl1, kl2 = KL[ν[1]], KL[ν[2]]
      if kl1.l != kl2.l
         keep[i] = false
      end
   end
   return Nu[keep]
end

function filter_tuples(KL, Nu, ::Val{3})  # 4B version
   keep = fill(true, length(Nu))
   for (i, ν) in enumerate(Nu)
      l1, l2, l3 = KL[ν[1]].l, KL[ν[2]].l, KL[ν[3]].l
      if !( (abs(l1-l2) <= l3 <= l1+l2) && iseven(l1+l2+l3) )
         keep[i] = false
      end
   end
   return Nu[keep]
end

# general tuple filter: !(abs(j1-j2) <= j3 <= j1 + j2)
#   for all (j1, j2, j3) ⊂ (l1, ..., ln)

function generate_LK_tuples(deg, wY::Real, bo; filter=true)
   # all possible (k, l) pairs
   allKL, degs = generate_LK(deg, wY)

   # the first iterm is just (0, ..., 0)
   # we can choose (k1, l1), (k2, l2) ... by indexing into allKL
   Nu = []
   _deg(ν) = maximum(ν) <= length(allKL) ? sum( allKL[n].deg for n in ν ) : Inf
   # Now we start incrementing until we hit the maximum degree
   # while retaining the ordering ν₁ ≤ ν₂ ≤ …
   lastidx = 0
   ν = MVector(ones(Int, bo)...)
   while true
      # we want to increment `curindex`, but if we've reach the maximum degree
      # then we need to move to the next index down

      # if the current tuple ν has admissible degree ...
      if _deg(ν) <= deg
         # ... then we add it to the stack  ...
         push!(Nu, SVector(ν))
         # ... and increment it
         lastidx = bo
         ν[lastidx] += 1
      else
         # we have overshot, _deg(ν) > deg; we must go back down, by
         # decreasing the index at which we increment
         if lastidx == 1
            break
         end
         ν[lastidx-1:end] .= ν[lastidx-1] + 1
         lastidx -= 1
      end
   end
   if filter; Nu = filter_tuples(allKL, Nu, Val(bo)); end
   return allKL, [ν for ν in Nu]
end


# -------------------------------------------------------------
#       define the basis itself
# -------------------------------------------------------------

"""
`struct SHIPBasis` : the main type around eveything in `SHIPs.jl` revolves;
it implements a permutation and rotation invariant basis.

### Developer Docs

* `deg` : total degree (to be generalised)
* `wY` : relative weighting of total degree definition; a (k,l) pair has total degree `deg(k,l) = k + wY * l`
* `J` : `TransformedJacobi` basis set for the `r`-component
* `SH` : spherical harmonics basis set for the `R̂`-component
* `KL` : list of all admissible `(k,l)` tuples
* `Nu` : a ν ∈ `Nu` specifies a basis function B_ν = ∑_m ∏_i A_νᵢm (details see `README.md`)
* `A, dA` : buffers for precomputing the `A_klm` functions
* `firstA` : same length as `KL`; each `(k,l) = KL[i]` has `2l+1` A_klm-functions associated which will be stored in the `A` buffer, the first of these is stored as `A[firstA[i]]`.
"""
struct SHIPBasis{BO, T, TJ}
   deg::Int
   wY::T
   J::TJ
   SH::SHBasis{T}
   KL::Vector{NamedTuple{(:k, :l, :deg),Tuple{Int,Int,T}}}
   Nu::Vector{SVector{BO, Int}}
   # ------ temporary storage arrays | TODO: create a `temp` function?
   A::Vector{Complex{T}}
   dA::Vector{JVec{Complex{T}}}
   BJ::Vector{T}
   dBJ::Vector{T}
   BSH::Vector{Complex{T}}
   dBSH::Vector{JVec{Complex{T}}}
   cg::ClebschGordan{T}
   # --------
   firstA::Vector{Int}   # indexing into A
   valBO::Val{BO}
end

length_A(deg, wY) = sum( sizeY( floor(Int, (deg - k)/wY) ) for k = 0:deg )

# this could become and allox_temp
alloc_A(deg, wY) = zeros(ComplexF64, length_A(deg, wY))
alloc_dA(deg, wY) = zeros(SVec3{ComplexF64}, length_A(deg, wY))

function _firstA(KL)
   idx = 1
   firstA = zeros(Int, length(KL) + 1)
   for i = 1:length(KL)
      firstA[i] = idx
      idx += 2 * KL[i].l + 1
   end
   firstA[end] = idx
   return firstA
end

function SHIPBasis(bo::Integer, deg::Integer, wY::Real, trans, p, rl, ru; filter=true)
   # r - basis
   maxP = deg
   J = rbasis(maxP, trans, p, rl, ru)
   # R̂ - basis
   maxL = floor(Int, deg / wY)
   SH = SHBasis(maxL)
   # get the basis specification
   allKL, Nu = generate_LK_tuples(deg, wY, bo; filter=filter)
   # allocate space for the A array
   A = alloc_A(deg, wY)
   dA = alloc_dA(deg, wY)
   # compute the (l,k) -> indexing into A information
   firstA = _firstA(allKL)
   @assert firstA[end] == length(A)+1
   # precompute the Clebsch-Gordan coefficients
   cg = ClebschGordan(maxL)
   # putting it all together ...
   return SHIPBasis(deg, wY, J, SH, allKL, Nu, A, dA,
                    alloc_B(J), alloc_dB(J), alloc_B(SH), alloc_dB(SH),
                    cg,  firstA, Val(bo))
end


bodyorder(ship::SHIPBasis{BO}) where {BO} = BO

length_B(ship::SHIPBasis{BO}) where {BO} = length(ship.Nu)

alloc_B(ship::SHIPBasis) = zeros(Float64, length_B(ship))
alloc_dB(ship::SHIPBasis) = zeros(SVec3{Float64}, length_B(ship))



# -------------------------------------------------------------
#       precompute the A arrays
# -------------------------------------------------------------

function precompute_A!(ship::SHIPBasis, Rs::AbstractVector{JVecF})
   fill!(ship.A, 0.0)
   for (iR, R) in enumerate(Rs)
      # evaluate the r-basis and the R̂-basis for the current neighbour at R
      eval_basis!(ship.BJ, ship.J, norm(R))
      eval_basis!(ship.BSH, ship.SH, R)
      # add the contributions to the A_klm; the indexing into the
      # A array is determined by `ship.firstA` which was precomputed
      for ((k, l), iA) in zip(ship.KL, ship.firstA)
         for m = -l:l
            @inbounds ship.A[iA+l+m] += ship.BJ[k+1] * ship.BSH[index_y(l, m)]
         end
      end
   end
   return nothing
end


# TODO
# function precompute_dA!(ship::SHIPBasis, Rs::AbstractVector{JVecF})


# -------------------------------------------------------------
#       Evaluate the actual basis functions
# -------------------------------------------------------------

"""
return kk, ll, mrange
where kk, ll is BO-tuples of k and l indices, while mrange is a
cartesian range over which to iterate to construct the basis functions

(note: this is tested for correcteness and speed)
"""
function _klm(ν::SVector{BO, T}, KL) where {BO, T}
   kk = SVector( ntuple(i -> KL[ν[i]].k, BO) )
   ll = SVector( ntuple(i -> KL[ν[i]].l, BO) )
   mrange = CartesianIndices(ntuple( i -> -ll[i]:ll[i], (BO-1) ))
   return kk, ll, mrange
end


"""
return the coefficients derived from the Clebsch-Gordan coefficients
that guarantee rotational invariance of the B functions
"""
function _Bcoeff(ll::SVector{BO, Int}, mm::SVector{BO, Int}) where {BO}
   @error("general case of B-coefficients has not yet been implemented")
end

function _Bcoeff(ll::SVector{2, Int}, mm::SVector{2, Int}, cg)
   @assert(mm[1] + mm[2] == 0)
   return (-1)^(mm[1])
end

function _Bcoeff(ll::SVector{3, Int}, mm::SVector{3, Int}, cg)
   @assert(mm[1] + mm[2] + mm[3] == 0)
   c = (ll[1], mm[1], ll[2], mm[3], ll[3], mm[3])
   w3j = (-1)^(ll[1]-ll[2]-mm[3]) / sqrt(2*ll[3]+1)
   return w3j
end


function eval_basis!(B, ship::SHIPBasis, Rs::AbstractVector{JVecF})
   precompute_A!(ship, Rs)
   KL = ship.KL
   for (idx, ν) in enumerate(ship.Nu)
      kk, ll, mrange = _klm(ν, KL)
      # b will eventually become B[idx], but we keep it Complex for now
      # so we can do a sanity check that it is in fact real.
      b = zero(ComplexF64)
      for m1 in mrange    # this is a cartesian loop over BO-1 indices
         mN = - sum(Tuple(m1))   # the last m-index is such that \sum mm = 0 (see paper!)
         if abs(mN) > ll[end]    # skip any m-tuples that aren't admissible
            continue
         end
         # compute the symmetry prefactor from the CG-coefficients
         mm = SVector(Tuple(m1)..., mN)
         C = _Bcoeff(ll, mm, ship.cg)
         bm = one(ComplexF64) * C
         for (i, (k, l, m)) in enumerate(zip(kk, ll, mm))
            # this is the indexing convention used to construct A
            #    (feels brittle - maybe rethink it and write a function for it)
            i0 = ship.firstA[ν[i]]
            bm *= ship.A[i0 + l + m]
         end
         b += bm
      end
      # two little sanity checks
      if b == 0.0
         @warn("B[idx] == 0!")
      end
      if abs(imag(b) / abs(b)) > 1e-10
         @warn("b/|b| == $(b/abs(b))")
      end
      B[idx] = real(b)
   end
   return B
end

# TODO
# function eval_basis_d!(B, ship::SHIPBasis, Rs::AbstractVector{JVecF})
