using ACE, Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using ACE.Testing 
using JuLIP: evaluate, evaluate_d
using ACE.Transforms: multitransform, transform, transform_d

##

@info("Testing Multi-Transform")

transforms = Dict(
     (:Fe, :C) => PolyTransform(2, (rnn(:Fe)+rnn(:C)) / 2), 
     (:C, :Al) => PolyTransform(2, (rnn(:Al)+rnn(:C)) / 2), 
     (:Fe, :Al) => PolyTransform(2, (rnn(:Al)+rnn(:Fe)) / 2), 
     (:Fe, :Fe) => PolyTransform(2, rnn(:Fe)), 
     (:Al, :Al) => PolyTransform(2, rnn(:Al)), 
     (:C, :C) => PolyTransform(2, rnn(:C)), 
     (:Al, :Fe) => PolyTransform(2, (rnn(:Al)+rnn(:Fe)) / 2 + 1)  
   )

zFe = AtomicNumber(:Fe)
zC = AtomicNumber(:C)
zAl = AtomicNumber(:Al)

trans = multitransform(transforms)

##

@info("Checking that transform, transform_d select the correct t")
for f in [transform, transform_d], z in (zFe, zC, zAl), z0 in (zFe, zC, zAl)
   z == zAl && z0 == zFe && continue    
   for ntest = 1:5 
      r = 1 + rand()
      print_tf(@test (f(trans, r, z, z0) 
                        == f(PolyTransform(2, (rnn(z)+rnn(z0)) / 2), r)))
   end
end
println() 

##

@info("Checking the non-symmetric case")
for f in [transform, transform_d]
   r = 1 + rand()
   println_slim(@test (f(trans, r, zAl, zFe) == 
            f(PolyTransform(2, (rnn(:Al)+rnn(:Fe)) / 2 + 1), r)))
end

## now try it again but with the affine transformation added on 

@info("check that with the AffineT the mapping goes to -1,1")
rin = 0.0
rcut = 5.0
trans = multitransform(transforms, rin = rin, rcut = rcut)

xmin = 1e30; xmax = - 1e30 
for ntest = 1:100
   r = rin + rand() * (rcut - rin)
   z, z0 = (rand([zFe, zC, zAl], 2)...,)
   x = transform(trans, r, z, z0)
   global xmin, xmax 
   xmin = min(x, xmin); xmax = max(x, xmax)
   print_tf(@test (abs(x) <= 1))
end
println() 
@show xmin, xmax 

##

@info("same for a flexible cutoff")


# define cutoffs as (rin, rcut) pairs. If (S1, S2) and (S2, S1) are both 
# specified then the cutoff is non-symmetric, If only one is specified, then 
# it will be symmetric 
cutoffs = Dict(
   (:Fe, :C) => (1.5, 5.0), 
   (:C, :Al) => (0.7, 6.0), 
   (:Fe, :Al) => (2.2, 4.5), 
   (:Fe, :Fe) => (2.0, 5.0), 
   (:Al, :Al) => (2.0, 5.0), 
   (:C, :C) => (1.5, 5.2), 
   (:Al, :Fe) => (1.5, 5.0)  )

trans2 = multitransform(transforms, cutoffs=cutoffs)

xmin = 1e30; xmax = - 1e30 
for ntest = 1:100
   local rin, rcut 
   z, z0 = (rand([zFe, zC, zAl], 2)...,)
   s, s0 = chemical_symbol.((z, z0))
   rin, rcut = try 
      cutoffs[(s, s0)]
   catch 
      cutoffs[(s0, s)]
   end 
   r = rin + rand() * (rcut - rin)
   x = transform(trans2, r, z, z0)
   global xmin, xmax 
   xmin = min(x, xmin); xmax = max(x, xmax)
   print_tf(@test (abs(x) <= 1))
end
println() 
@show xmin, xmax 

##

@info("check at the boundaries")
for ((s, s0), (rin, rcut)) in cutoffs 
   z, z0 = AtomicNumber.((s, s0))
   print_tf( @test( transform(trans2, rin, z, z0) ≈ -1 ) )
   print_tf( @test( transform(trans2, rcut, z, z0) ≈ 1 ) )
end

##

@info("      test (de-)dictionisation")
println_slim(@test read_dict(write_dict(trans2)) == trans2)
println_slim(@test all(JuLIP.Testing.test_fio(trans)))
println_slim(@test all(JuLIP.Testing.test_fio(trans2)))

##


verbose = true
maxdeg = 10

@info("Testing TransformedPolys")

B = transformed_jacobi(maxdeg, trans2; pcut = 2)

# this should fail 
@info("should not evaluate without species info:")
println_slim(@test (try
                  evaluate(B, 1+rand())
                  false 
               catch
                  true 
               end))

@info("evaluate with species info:")
println_slim(@test (try
                     evaluate(B, 1+rand(), zFe, zC)
                  true 
               catch
                  false 
               end))

##

@info("some random finite-difference tests")
for ntest = 1:30 
   r = 2 + rand() 
   z, z0 = (rand([zFe, zC, zAl], 2)...,)
   P = evaluate(B, r, z, z0)
   dP = evaluate_d(B, r, z, z0)
   U = randn(length(P))
   F = t -> dot( evaluate(B, t[1], zFe, zC), U )
   dF = t -> [ dot( evaluate_d(B, t[1], zFe, zC), U ) ] 
   print_tf(@test all( JuLIP.Testing.fdtest(F, dF, [r], verbose=false) ))
end


## read / write 

@info("Testing FIO")
println_slim(@test all( JuLIP.Testing.test_fio(trans) ))


## Create a symmetric basis with this transform 
# and just check that it actually evaluates ok. 

@info("Test whether we can evaluate a symmetric basis with this thing")
maxdeg = 8
N = 3
Pr = transformed_jacobi(maxdeg, trans2; pcut = 2)
D = SparsePSHDegree()
P1 = BasicPSH1pBasis(Pr; species = [:Fe, :Al, :C], D = D)
pibasis = PIBasis(P1, N, D, maxdeg)
rpibasis = RPIBasis(P1, N, D, maxdeg)

using StaticArrays
Nat = 15
z0 = zFe 
Zs = rand([zFe, zC, zAl], Nat)
randr = () -> (r = 2.3 + 2.5 * rand(); x = randn(SVector{3, Float64}); x/r)
Rs = [ randr() for _=1:Nat ]

B1 = evaluate(rpibasis, Rs, Zs, z0)
dB1 = evaluate_d(rpibasis, Rs, Zs, z0)

# seems to work ok - TODO : implemeant an actual test that checks
# what really goes on i.e. how the basis changes as the chemical environment 
# changes? I'm not sure what to test though...
