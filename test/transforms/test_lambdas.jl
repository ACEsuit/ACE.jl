

using ACE, Test, StaticArrays, BenchmarkTools, ACEbase 
using ACE: read_dict, write_dict, 
           evaluate, evaluate_d, evaluate_dd
using ACEbase.Testing: println_slim, print_tf, fdtest, test_fio 
using LinearAlgebra: norm 

# TODO: move this into ACEbase.Testing 
function save_load(obj)
   D = write_dict(obj)
   tmpf = tempname() * ".json"
   ACEbase.FIO.save_dict(tmpf, D)
   D2 = ACEbase.FIO.load_dict(tmpf)
   return read_dict(load_dict(tmpf))
end

##

@info("Testing Legible Lambdas")
ff = [ (λ("  r -> exp(-r)"), r -> exp(-r), () -> 3 * rand()), 
       (λ("  r -> 1 / (1 + r)^3"), r -> 1 / (1 + r)^3, () -> 3 * rand()), 
       (λ(" rr -> norm(rr)"), rr -> norm(rr), () -> randn(SVector{3, Float64})), 
       (λ(" rr -> exp(-sum(abs2, rr))"), rr -> exp(-sum(abs2, rr)), () -> randn(SVector{3, Float64})), 
       ]

for (f, g, rndx) in ff 
   show(f);  println() 
   println_slim(@test all(test_fio(f; warntype=false)))
   xx = [ rndx() for _=1:30 ]
   println_slim( @test f.(xx) ≈ g.(xx) )
   f1 = save_load(f)
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

@info("Confirm zero allocations and performance")
tmin = minimum( (@benchmark $g(($rndx)())).times )
for h in [f, f1]
   bm = @benchmark evaluate($h, ($rndx)())
   println(@test bm.allocs == 0 && bm.memory == 0)
   println(@test (minimum(bm.times) <= 1.2 * tmin)) 
end
