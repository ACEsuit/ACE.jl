
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



using JuLIP, ACE, LinearAlgebra

deg = 10
trans = ACE.IdTransform()
J = ACE.TransformedJacobi(deg, trans, 0.5, 3.0)
ship4 = SHIPBasis(SparseSHIP(3, 10; wL = 1.5), J)
J = ship4.J

nargs = 3
function testf(Rs)
   f = 1.0
   for R in Rs
      b = ACE.evaluate(J, norm(R))
      f *= b[3]
   end
   return f
end

# testf(Rs) = prod(norm.(Rs).^2)

ACE.Exp.determine_order(testf, nargs, ship4)
