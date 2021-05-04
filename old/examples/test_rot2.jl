
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


using JuLIP
using ACE, ACE.SphericalHarmonics, StaticArrays, LinearAlgebra,
      Combinatorics
using ACE: _mrange
using ACE.Rotations

##
lmax = 3
A = ACE.Rotations.Rot3DCoeffs(Float64)
for l1 = 0:lmax, l2=0:lmax, l3=0:lmax, l4=0:lmax
   global A
   ll = SVector(l1, l2, l3, l4)
   Uslim = ACE.Rotations.compute_Al(A, ll, ordered=true)
   Uall = ACE.Rotations.compute_Al(A, ll, ordered=false)
   if rank(Uslim) != rank(Uall)
      @show ll, rank(Uslim), rank(Uall)
   end
end

<<<<<<< HEAD
=======
ll = SVector(1,1,1,0)
U = ACE.Rotations.compute_Al(A, ll, ordered=false)


>>>>>>> 14ee199fe32862737cd306ae0033dac552f4f512
## -------------

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ], zeros(Int16, N)
randR(N, syms) = randR(N)[1], rand( Int16.(atomic_number.(syms)), N )

trans = PolyTransform(2, 1.0)
cutf = PolyCutoff2s(2, 0.5, 3.0)
ship4 = SHIPBasis(SparseSHIP(4, 10; wL = 1.5), trans, cutf)

idx = (length(ship4)-2):length(ship4)
G = zeros(1000, 3)
tmp = JuLIP.alloc_temp(ship4, 5)
for n = 1:1000
   R, Z = randR(5)
   G[n, :] = evaluate(ship4, R, Z, 0)[idx]
end

rank(G)
ship4.NuZ[end][end]
ACE._get_ll(ship4.KL[end], ship4.NuZ[end][end])
# i.e. ll = (1,1,1,1)



# --------------------

const SH = ACE.SphericalHarmonics.SHBasis(10)

ip(t1, t2) = (t1 == t2)

ll1 = [1,2,2]
mm1 = [-1,2,1]
ll2 = [1,2,2]
mm2 = [-1,1,2]

ipsym = 0
for σ in permutations(1:3)
   global ipsym += ip([ll1; mm1], [ll2[σ]; mm2[σ]])
end
ipsym



ll = SVector(4,4,4,4)
CC, KK = RotationCoeffs.compute_Al_symm(ll)
DK = Dict([ kk => ik for (ik, kk) in enumerate(KK)]...)
sumCC = zeros(size(CC))
for p in permutations(1:4)
   global sumCC
   # kk = KK[i] then a_kk,j = CC[i,j]
   # kk[p] = KK[ip] then CCp[ip,:] = CC[i,:]
   CCp = zeros(size(CC))
   for i = 1:size(CC,1)
      kk = KK[i]
      # kk[p] is the permutation of p
      ip = DK[SVector(kk[p] ...)]
      CCp[ip, :] = CC[i, :]
   end
   sumCC += CCp
end

rank(CC)
rank(sumCC)
svdvals(sumCC)



function Alkm_old(ll::SVector{2}, mm, kk, cg)
   if ll[1] != ll[2] ||  sum(mm) != 0 || sum(kk) != 0
      return 0.0
   end
   return 8 * pi^2 / (2*ll[1]+1) * (-1)^(mm[1]-kk[1])
end


function Alkm_old(ll::SVector{3}, mm, kk, cg)
   if sum(mm) != 0 || sum(kk) != 0
      return 0.0
   end
   return 8 * pi^2 / (2*ll[3]+1) * (-1)^(mm[3]-kk[3]) *
          cg(ll[1], mm[1], ll[2], mm[2], ll[3], -mm[3]) *
          cg(ll[1], kk[1], ll[2], kk[2], ll[3], -kk[3])
end

function Alkm_old(ll::SVector{4}, mm, kk, cg)
   Alkm = 0.0
   jlo = max(abs(ll[1]-ll[2]), abs(ll[3]-ll[4]))
   jhi = min(ll[1]+ll[2], ll[3]+ll[4])
   for j = jlo:jhi
      if (abs(mm[1]+mm[2]) > j) || (abs(mm[3]+mm[4]) > j) ||
         (abs(kk[1]+kk[2]) > j) || (abs(kk[3]+kk[4]) > j)
         continue
      end
      Alkm += 8 * pi^2 * (-1)^(mm[1]+mm[2]-kk[1]-kk[2]) / (2*j+1) *
                    cg(ll[1], mm[1], ll[2], mm[2], j, mm[1]+mm[2]) *
                    cg(ll[3], mm[3], ll[4], mm[4], j, mm[3]+mm[4]) *
                    cg(ll[1], kk[1], ll[2], kk[2], j, kk[1]+kk[2]) *
                    cg(ll[3], kk[3], ll[4], kk[4], j, kk[3]+kk[4])
   end
   return Alkm
end

##
A = RotationCoeffs.Rot3DCoeffs(5, 12)
cg = ClebschGordan(12)

## len-2
l2a = SVector(2,3)
l2b = SVector(2,2)
m2 = SVector(-1,1)
k2 = SVector(2,-2)
Alkm_old(l2a, m2, k2, cg)
A(l2a, m2, k2)
Alkm_old(l2b, m2, k2, cg)
A(l2b, m2, k2)

## len-3
l3 = SVector(3,2,3)
m3 = SVector(-2,1,1)
k3 = SVector(0,2,-2)
Alkm_old(l3, m3, k3, cg)
A(l3, m3, k3)

##  len-4
l4 = SVector(3,3,3,3)
m4 = SVector(-2,1,3,-2)
k4 = SVector(0,3,-2,-1)
Alkm_old(l4, m4, k4, cg)
A(l4, m4, k4)

A4 = A.vals[4]

ll = SVector(5,3,2,2)
@time RotationCoeffs.compute_Al(A, ll)

ll = SVector(3,2,4)
A = RotationCoeffs.Rot3DCoeffs(5, 12)
ll = SVector(3,2)
@time Cl = RotationCoeffs.compute_Al(A, ll)
ll = SVector(4,3,3,2)
@time Cl = RotationCoeffs.compute_Al(A, ll)
@time Cold = compute_Ckm_old(ll)

Ul = svd(Cl).U[:, 1:5]
Uo = svd(Cold).U[:, 1:5]

rank([Ul Uo])


ll = SVector(2, 2, 2, 2, 2)
Cl = RotationCoeffs.compute_Al(A, ll)
rank(Cl)
Matrix(svd(Cl).U)
A.vals[5]

function compute_Ckm_old(ll::SVector{4})
   cg = ClebschGordan(sum(ll))

   len = 0
   for mm in _mrange(ll)
      len += 1
   end
   @show len

   Ckm = zeros(len, len)

   for (im, mm) in enumerate(_mrange(ll)), (ik, kk) in enumerate(_mrange(ll))
      jlo = max(abs(ll[1]-ll[2]), abs(ll[3]-ll[4]))
      jhi = min(ll[1]+ll[2], ll[3]+ll[4])
      for j = jlo:jhi
         if (abs(mm[1]+mm[2]) > j) || (abs(mm[3]+mm[4]) > j) ||
            (abs(kk[1]+kk[2]) > j) || (abs(kk[3]+kk[4]) > j)
            continue
         end
         cg1 = cg(ll[1], mm[1], ll[2], mm[2], j, mm[1]+mm[2])
         cg2 = cg(ll[3], mm[3], ll[4], mm[4], j, mm[3]+mm[4])
         cg3 = cg(ll[1], kk[1], ll[2], kk[2], j, kk[1]+kk[2])
         # @show ll[3], kk[3], ll[4], kk[4], j, kk[3]+kk[4]
         cg4 = cg(ll[3], kk[3], ll[4], kk[4], j, kk[3]+kk[4])
         Ckm[ik,im] += 8 * pi^2 * (-1)^(mm[1]+mm[2]-kk[1]-kk[2]) / (2*j+1) *
                       cg(ll[1], mm[1], ll[2], mm[2], j, mm[1]+mm[2]) *
                       cg(ll[3], mm[3], ll[4], mm[4], j, mm[3]+mm[4]) *
                       cg(ll[1], kk[1], ll[2], kk[2], j, kk[1]+kk[2]) *
                       cg(ll[3], kk[3], ll[4], kk[4], j, kk[3]+kk[4])
      end
   end
   return Ckm
end

##


##

# CASE 1
ll1 = SVector(2,1,1,2)
Ckm = compute_Ckm(ll1)
@show rank(Ckm)
svdf = svd(Ckm)
@show svdf.S[1:5]
for i = 1:3
   @info("V[:,$i]")
   @show round.(svdf.Vt[i,:], digits=2)
end

# CASE 2
ll2 = SVector(2,3,4,3)
Ckm = compute_Ckm(ll2)
@show rank(Ckm)
@show svdvals(Ckm)[1:8]

# CASE 3
ll3 = SVector(5,4,4,3)
Ckm = compute_Ckm(ll3)
@show rank(Ckm)
@show svdvals(Ckm)[1:10]


# MAIN EXAMPLE:
ll = SVector(1,1,1,1)
Ckm = compute_Ckm(ll)
Ckm_sym = [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 15.7914 0.0 -7.89568 -7.89568 2.63189 5.26379 2.63189 0.0 -7.89568 -7.89568 5.26379 10.5276 5.26379 -7.89568 -7.89568 0.0 2.63189 5.26379 2.63189 -7.89568 -7.89568 0.0 15.7914 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 -7.89568 0.0 10.5276 -2.63189 -7.89568 -2.63189 5.26379 0.0 -2.63189 10.5276 -2.63189 -5.26379 -2.63189 10.5276 -2.63189 0.0 5.26379 -2.63189 -7.89568 -2.63189 10.5276 0.0 -7.89568 0.0 0.0; 0.0 0.0 -7.89568 0.0 -2.63189 10.5276 5.26379 -2.63189 -7.89568 0.0 10.5276 -2.63189 -2.63189 -5.26379 -2.63189 -2.63189 10.5276 0.0 -7.89568 -2.63189 5.26379 10.5276 -2.63189 0.0 -7.89568 0.0 0.0; 0.0 0.0 2.63189 0.0 -7.89568 5.26379 15.7914 -7.89568 2.63189 0.0 5.26379 -7.89568 -7.89568 10.5276 -7.89568 -7.89568 5.26379 0.0 2.63189 -7.89568 15.7914 5.26379 -7.89568 0.0 2.63189 0.0 0.0; 0.0 0.0 5.26379 0.0 -2.63189 -2.63189 -7.89568 10.5276 -7.89568 0.0 -2.63189 -2.63189 10.5276 -5.26379 10.5276 -2.63189 -2.63189 0.0 -7.89568 10.5276 -7.89568 -2.63189 -2.63189 0.0 5.26379 0.0 0.0; 0.0 0.0 2.63189 0.0 5.26379 -7.89568 2.63189 -7.89568 15.7914 0.0 -7.89568 5.26379 -7.89568 10.5276 -7.89568 5.26379 -7.89568 0.0 15.7914 -7.89568 2.63189 -7.89568 5.26379 0.0 2.63189 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 -7.89568 0.0 -2.63189 10.5276 5.26379 -2.63189 -7.89568 0.0 10.5276 -2.63189 -2.63189 -5.26379 -2.63189 -2.63189 10.5276 0.0 -7.89568 -2.63189 5.26379 10.5276 -2.63189 0.0 -7.89568 0.0 0.0; 0.0 0.0 -7.89568 0.0 10.5276 -2.63189 -7.89568 -2.63189 5.26379 0.0 -2.63189 10.5276 -2.63189 -5.26379 -2.63189 10.5276 -2.63189 0.0 5.26379 -2.63189 -7.89568 -2.63189 10.5276 0.0 -7.89568 0.0 0.0; 0.0 0.0 5.26379 0.0 -2.63189 -2.63189 -7.89568 10.5276 -7.89568 0.0 -2.63189 -2.63189 10.5276 -5.26379 10.5276 -2.63189 -2.63189 0.0 -7.89568 10.5276 -7.89568 -2.63189 -2.63189 0.0 5.26379 0.0 0.0; 0.0 0.0 10.5276 0.0 -5.26379 -5.26379 10.5276 -5.26379 10.5276 0.0 -5.26379 -5.26379 -5.26379 15.7914 -5.26379 -5.26379 -5.26379 0.0 10.5276 -5.26379 10.5276 -5.26379 -5.26379 0.0 10.5276 0.0 0.0; 0.0 0.0 5.26379 0.0 -2.63189 -2.63189 -7.89568 10.5276 -7.89568 0.0 -2.63189 -2.63189 10.5276 -5.26379 10.5276 -2.63189 -2.63189 0.0 -7.89568 10.5276 -7.89568 -2.63189 -2.63189 0.0 5.26379 0.0 0.0; 0.0 0.0 -7.89568 0.0 10.5276 -2.63189 -7.89568 -2.63189 5.26379 0.0 -2.63189 10.5276 -2.63189 -5.26379 -2.63189 10.5276 -2.63189 0.0 5.26379 -2.63189 -7.89568 -2.63189 10.5276 0.0 -7.89568 0.0 0.0; 0.0 0.0 -7.89568 0.0 -2.63189 10.5276 5.26379 -2.63189 -7.89568 0.0 10.5276 -2.63189 -2.63189 -5.26379 -2.63189 -2.63189 10.5276 0.0 -7.89568 -2.63189 5.26379 10.5276 -2.63189 0.0 -7.89568 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 2.63189 0.0 5.26379 -7.89568 2.63189 -7.89568 15.7914 0.0 -7.89568 5.26379 -7.89568 10.5276 -7.89568 5.26379 -7.89568 0.0 15.7914 -7.89568 2.63189 -7.89568 5.26379 0.0 2.63189 0.0 0.0; 0.0 0.0 5.26379 0.0 -2.63189 -2.63189 -7.89568 10.5276 -7.89568 0.0 -2.63189 -2.63189 10.5276 -5.26379 10.5276 -2.63189 -2.63189 0.0 -7.89568 10.5276 -7.89568 -2.63189 -2.63189 0.0 5.26379 0.0 0.0; 0.0 0.0 2.63189 0.0 -7.89568 5.26379 15.7914 -7.89568 2.63189 0.0 5.26379 -7.89568 -7.89568 10.5276 -7.89568 -7.89568 5.26379 0.0 2.63189 -7.89568 15.7914 5.26379 -7.89568 0.0 2.63189 0.0 0.0; 0.0 0.0 -7.89568 0.0 -2.63189 10.5276 5.26379 -2.63189 -7.89568 0.0 10.5276 -2.63189 -2.63189 -5.26379 -2.63189 -2.63189 10.5276 0.0 -7.89568 -2.63189 5.26379 10.5276 -2.63189 0.0 -7.89568 0.0 0.0; 0.0 0.0 -7.89568 0.0 10.5276 -2.63189 -7.89568 -2.63189 5.26379 0.0 -2.63189 10.5276 -2.63189 -5.26379 -2.63189 10.5276 -2.63189 0.0 5.26379 -2.63189 -7.89568 -2.63189 10.5276 0.0 -7.89568 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 15.7914 0.0 -7.89568 -7.89568 2.63189 5.26379 2.63189 0.0 -7.89568 -7.89568 5.26379 10.5276 5.26379 -7.89568 -7.89568 0.0 2.63189 5.26379 2.63189 -7.89568 -7.89568 0.0 15.7914 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]

badI = [1,2,4,10, 18, 24, 26, 27]
goodI = setdiff(1:27, badI)
Ckm_sym = Ckm_sym[goodI, goodI]


display(Ckm_sym[:, [1,2,3]])
display(Ckm[:, [1,2,3]])

norm(Ckm_sym - Ckm, Inf)
svdvals(Ckm)
svdvals(Ckm_sym)

ll = SVector(2,1,1,1)
Ckm = compute_Ckm(ll)
rank(Ckm)
c1 = Ckm[:,1]
c2 = Ckm[:,3]
c1 /= norm(c1)
c2 /= norm(c2)
[sort(c1) sort(c2)]

U = svd(Ckm).U[:,1:2]
[ sort(U[:,1]) sort(U[:,2]) ]
U' * U

svd(Ckm)

for l = 1:6
   ll = SVector(l,l,l,l)
   Ckm = compute_Ckm(ll)
   @show l, rank(Ckm)
end

using Profile
A = RotationCoeffs.Rot3DCoeffs(5, 12)
ll = SVector(1,1,1,1,1)
ll = SVector(4,4,2,1,1)
@profile Al = RotationCoeffs.compute_Al(A, ll)

Profile.print()
