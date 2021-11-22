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
   xmin = min(x, xmin); xmax = max(x, xmax)
   print_tf(@test (abs(x) <= 1))
end
println() 
@show xmin, xmax 


##

# @info("      test (de-)dictionisation")
# println(@test read_dict(write_dict(trans)) == trans)

##

verbose = true
maxdeg = 10

@info("Testing TransformedPolys")

B = transformed_jacobi(maxdeg, trans, 5.0; pcut = 2)

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