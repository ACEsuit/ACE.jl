##

import ACE, ACEbase
using LinearAlgebra, StaticArrays, Test, Printf, ACE.Testing
using ACE: evaluate, evaluate_d, evaluate_ed, Trig1pBasis, Trig

##

@info(" --------- Trig Tests --------- ")
@info("Test conversion to angle")
for ntest = 1:30 
   θ = 2*π*rand() - π 
   rr = [cos(θ), sin(θ)]
   print_tf(@test( ACE._theta(rr) ≈ θ ))
end
println()

##

@info("test gradient of coordinate transform")
for ntest = 1:30
   rr = randn(2) # [cos(θ), sin(θ)]
   G = rr -> ACE._theta(rr)
   dG = rr -> ACE._theta_ed(rr)[2] |> Vector
   print_tf(@test( all( 
            ACEbase.Testing.fdtest(G, dG, rr; verbose=false) 
         )))
end
println()

##

@info("test evaluation of trig basis")

bE = Trig(10)

for ntest = 1:30
   L = bE.L
   θ = π * rand()
   E = [ exp(im * l * θ) for l = -L:L ]
   rr = [cos(θ), sin(θ)]
   print_tf( @test evaluate(bE, rr) ≈ E )
end
println() 

##

@info("test jacobian of trig basis ")
θ = π * rand()
rr = [cos(θ), sin(θ)]
E, dE = evaluate_ed(bE, rr)

for ntest = 1:30 
   rr = randn(2) 
   U = randn(length(E))
   F = rr -> ACE.contract(evaluate(bE, rr), U)
   dF = rr -> Vector(ACE.contract(evaluate_ed(bE, rr)[2], U))
   print_tf(@test( all(
          ACEbase.Testing.fdtest(F, dF, rr; verbose=false) 
      )))
end
println()

##
