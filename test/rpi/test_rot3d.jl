
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



A = ACE.Rotations3D.Rot3DCoeffs()
ACE.Rotations3D.basis(A, SVector(2, 2, 1, 1))

using ACE, LinearAlgebra

trans = PolyTransform(2, 1.0)
fcut = PolyCutoff1s(2, 0.1, 2.0)
basis = SHIPBasis(SparseSHIP(5, 8, 1.0), trans, fcut)

length(basis)

A = ACE.Rotations.Rot3DCoeffs(20)

for bo = 2:5
   ctr = 0
   println();
   @show bo
   for νz in basis.NuZ[bo]
      ν = νz.ν
      kk = getfield.(basis.KL[1][ν], :k)
      if norm(kk) != 0
         continue
      end
      ll = getfield.(basis.KL[1][ν], :l)
      U = ACE.Rotations.basis(A, ll)
      c = [ ACE._Bcoeff(ll, mm, A.cg) for mm in ACE._mrange(ll) ]
      def = norm(U * (U' * c) - c) > 1e-12
      if def > 1e-12
         @show ll, def
         ctr += 1
         if ctr > 10
            break
         end
      else
         print("✓")
      end
   end
end


using StaticArrays

for bo = 2:5
   println()
   @show bo
   ctr = 0
   cartrg = CartesianIndices( ntuple(_->0:3, bo) )
   for ill in cartrg
      ll = SVector(ill.I...)
      if issorted(ll) && sum(ll) <= 20
         U =  try
            ACE.Rotations.basis(A, ll)
         catch
            println()
            @show ll
            continue
         end
         c = [ ACE._Bcoeff(ll, mm, A.cg) for mm in ACE._mrange(ll) ]
         def = norm(U * (U' * c) - c)
         if def > 1e-12
            @show ll, def
            ctr += 1
            if ctr > 10
               break
            end
         else
            print("✓")
         end
      end
   end
end
