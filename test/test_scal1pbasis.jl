
@testset "Scalar1PBasis" begin 

##
using ACE
using Printf, Test, LinearAlgebra, StaticArrays
using ACE: evaluate, evaluate_d, Rn1pBasis, Ylm1pBasis,
      EuclideanVectorState, Product1pBasis
using Random: shuffle
using ACEbase.Testing: dirfdtest, fdtest, print_tf, test_fio

##

maxdeg = 10 
r0 = 1.0 
rcut = 3.0 
trans = trans = PolyTransform(1, r0)
bscal = ACE.scal1pbasis(:x, :k, maxdeg, trans, rcut)

ACE.evaluate(bscal, 1.2)
ACE.evaluate_d(bscal, 1.0)



##

@info("Test FIO")
println(@test(all(test_fio(bscal))))

##
end