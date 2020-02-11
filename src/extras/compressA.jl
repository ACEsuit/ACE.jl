
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



##
using Combinatorics, LinearAlgebra

using PoSH, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using JuLIP
using JuLIP: evaluate, evaluate_d
using JuLIP.Testing

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ], zeros(Int16, N), 0
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

##

function delete_rows(A::Matrix{T}, Idel) where {T}
   Idel = [sort(unique(Idel)); size(A,1)+1]
   B = Matrix{T}(undef, size(A,1) - length(Idel) + 1, size(A, 2))
   iB = 0
   iI = 1
   for iA = 1:size(A,1)
      # if iA == Idel[iI]
      #    iI += 1
      if !(iA in Idel)
         iB += 1
         B[iB, :] .= A[iA, :]
      end
   end
   return B
end

function remove_zeros!(ship::SHIP{T, NZ}; tol=1e-10) where {T, NZ}
   @assert NZ == 1
   Idel = sort(findall(abs.(ship.coeffs[1]) .< tol))
   ship2 = deepcopy(ship)
   deleteat!(ship2.coeffs[1], Idel)
   deleteat!(ship2.aalists[1].len, Idel)
   ship2.aalists[1].i2Aidx = delete_rows(ship2.aalists[1].i2Aidx, Idel)
   for (key, val) in ship2.aalists[1].zklm2i
      if val in Idel
         delete!(ship2.aalists[1].zklm2i, key)
      end
   end
   return ship2
end


function _eval_AA(ship::SHIP{T,NZ}, alist, aalist, Iaa, Rs) where {T, NZ}
   @assert NZ ==  1
   Zs = zeros(Int16, length(Rs))
   A = PoSH.alloc_A(alist)
   tmp = PoSH.alloc_temp(ship, length(Rs))
   PoSH.precompute_A!(A, tmp, alist, Rs, Zs, ship)
   AA = ones(Complex{T}, length(Iaa))
   for i = 1:length(Iaa)
      for α = 1:aalist.len[Iaa[i]]
         AA[i] *= A[ aalist.i2Aidx[Iaa[i], α] ]
      end
   end
   return AA
end

function _sample_AA(ship::SHIP{T,NZ}, alist, aalist, Iaa,
                    nsamples=length(Iaa)*10) where {T, NZ}
   N = aalist.len[Iaa[1]]
   AA = zeros(Complex{T}, nsamples, length(Iaa))
   for ns = 1:nsamples
      Rs = PoSH.Utils.rand(ship.J, N)
      AA[ns, :] = _eval_AA(ship, alist, aalist, Iaa, Rs)
   end
   return AA
end


function _filter_AA!(ship, coeffs, alist, aalist, Iaa; tol = 1e-7)
   # if length(Iaa) == 1 then there is nothing to do
   if length(Iaa) == 1; return; end
   # otherwise start by constructing a representation of the basis functios
   # we are trying to compress
   AA = _sample_AA(ship, alist, aalist, Iaa)
   Svd = svd(AA)
   izeros = findall(Svd.S .< maximum(Svd.S) * tol)
   # if there are no zeros then we are also done
   if length(izeros) == 0; return; end

   # keep just one linear combination that give us zero...
   iz = izeros[1]
   vz = Svd.V[:, iz]
   # vz[amax] is the most stable basis function to throw away due to
   #   AA[:, amax] = - ∑_{a≂̸amax} vz[a] / vz[amax] AA[:,a]
   amax = findmax(abs.(vz))[2]
   # now distribute the coeffs[Iaa[amax]] coefficient to the
   # coeffs[Iaa[a]] coefficients
   for a = 1:length(Iaa)
      if a != amax
         coeffs[Iaa[a]] -= vz[a] / vz[amax] * coeffs[Iaa[amax]]
      end
   end
   # ... and remove the Iaa[amax] basis function entirely by setting  the
   # coefficient to zero
   coeffs[Iaa[amax]] = 0
   # now also delete it from the index list Iaa
   deleteat!(Iaa, amax)
   # now call the filtering recursively
   return _filter_AA!(ship, coeffs, alist, aalist, Iaa; tol=tol)
end

permute_AA(t, σ) = ( t[1][σ], t[2][σ], t[3][σ], t[4][σ] )

function compressA(ship::SHIP{T, NZ};
                   ) where{T, NZ}
   @assert NZ == 1   # TODO: obviously we need to turn this off

   alist = ship.alists[1]
   aalist = ship.aalists[1]
   coeffs = copy(ship.coeffs[1])

   all_zzkkllmm = collect(keys(aalist.zklm2i)) # TODO: should use a Set datastructure here
   while length(all_zzkkllmm) > 0
      zzkkllmm0 = all_zzkkllmm[1]
      N = length(zzkkllmm0[1])
      Iaa = Int[ aalist.zklm2i[zzkkllmm0] ]
      deleteat!(all_zzkkllmm, 1)
      # find all "permutation-equivalent" basis functions
      for σ in permutations(1:N)
         σs = SVector(σ...)
         σ_zzkkllmm0 = permute_AA(zzkkllmm0, σ)
         if haskey(aalist.zklm2i, σ_zzkkllmm0)
            iσ = aalist.zklm2i[σ_zzkkllmm0]
            #  TODO: we could check whether the coefficient is zero
            if !(iσ in Iaa)
               # append this basis function index to Iaa to be part of the
               # current compression cycle
               push!(Iaa, iσ)
               # then delete this basis function from the list
               # all_zzkkllmm so that we don't try to account for it a
               # second time.
               Idel = findall(all_zzkkllmm .== Ref(σ_zzkkllmm0))
               @assert length(Idel) == 1
               deleteat!(all_zzkkllmm, Idel)
            end
         end
      end

      # now that we've found a permutation-equivalent group of basis
      # functions we can filter it. This will only rewrite the coeffs array,
      # by setting some coefficients to zero while changing some other
      # coefficients.
      _filter_AA!(ship, coeffs, alist, aalist, Iaa)
   end

   # we've now filtered all permutation-equivalent basis groups. We now
   # remove all zeros.
   shipnew = deepcopy(ship)
   shipnew.coeffs[1][:] = coeffs
   return remove_zeros!(shipnew)
end


@info("Generate a basis")
basis = PoSH.Utils.TestBasis(6,13)
length(basis)
@info("Convert to the SHIP")
ship = SHIP(basis, randcoeffs(basis))
@info("Compress the coefficients")
shipc = compressA(ship)
@info("check the lengths")
length(ship.coeffs[1])
length(shipc.coeffs[1])

# need to check now that the two evaluate the same.
@info("Check correctness")
for n = 1:30
   Rs = PoSH.Utils.rand(ship.J, 10)
   Zs = zeros(Int16, length(Rs))
   z0 = 0
   s1 = evaluate(ship, Rs, Zs, z0)
   s2 = evaluate(shipc, Rs, Zs, z0)
   print_tf(@test s1 ≈ s2)
end


@info("Evaluation test")
Rs = PoSH.Utils.rand(ship.J, 30)
Zs = zeros(Int16, length(Rs))
z0 = 0
tmp = PoSH.alloc_temp(ship, length(Rs))
tmp_d = PoSH.alloc_temp_d(ship, length(Rs))
dEs = zeros(JVecF, 30)
@info("  evaluate!:")
@btime PoSH.evaluate!($tmp, $ship, $Rs, $Zs, $z0)
@btime PoSH.evaluate!($tmp, $shipc, $Rs, $Zs, $z0)
@info("  evaluate_d!:")
@btime PoSH.evaluate_d!($dEs, $tmp_d, $ship, $Rs, $Zs, $z0)
@btime PoSH.evaluate_d!($dEs, $tmp_d, $shipc, $Rs, $Zs, $z0)
