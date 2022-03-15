

using ACE, Test, StaticArrays, BenchmarkTools
using ACE: read_dict, write_dict, 
           evaluate, evaluate_d, evaluate_dd
using ACEbase.Testing: println_slim, print_tf, fdtest
using LinearAlgebra: norm 

##

@info("Testing Legible Lambdas")
ff = [ ((@λ r -> exp(-r)), r -> exp(-r), () -> 3 * rand()), 
       ((@λ r -> 1 / (1 + r)^3), r -> 1 / (1 + r)^3, () -> 3 * rand()), 
       ((@λ rr -> norm(rr)), rr -> norm(rr), () -> randn(SVector{3, Float64})), 
       ((@λ rr -> exp(-sum(abs2, rr))), rr -> exp(-sum(abs2, rr)), () -> randn(SVector{3, Float64})), 
       ]

for (f, g, rndx) in ff 
   show(f);  println() 
   xx = [ rndx() for _=1:30 ]
   println_slim( @test f.(xx) ≈ g.(xx) )
   D = write_dict(f)
   f1 = read_dict(D)
   println_slim( @test f.(xx) ≈ f1.(xx) )
   for ntest = 1:30 
      x = rand() * 3 
      print_tf(@test all(fdtest(x -> evaluate(f, x), x -> evaluate_d(f, x), x; verbose=false)))
   end
   println() 
end


## 

@info("Performance test") 

f, g, rndx = ff[end]
f1 = read_dict(write_dict(f))

print("      Raw Anon:"); @btime $g(($rndx)())
print(" LegibleLambda:"); @btime evaluate($f, ($rndx)())
print("  Deserialized:"); @btime evaluate($f1, ($rndx)())

##