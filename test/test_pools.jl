

using ACE, BenchmarkTools

pool = ACE.ArrayCache{Float64}()

##


function runn(x, N, len)
   @assert length(x) >= len
   fill!(x, 0)
   for n = 1:N, i = 1:len 
      x[i] += i/n
   end 
   return nothing 
end

x = rand(1000)
@btime runn($x, 1000, 1000) 

##

function runn2(cache, N, len)
   x = ACE.acquire!(cache, len)
   fill!(x.A, 0)
   for n = 1:N, i = 1:len 
      x[i] += i/n
   end 
   ACE.release!(x)
   return nothing 
end

@btime runn2($pool, 1000, 1000) 

##

N = 1000 

for len in [1000, 300, 100, 30, 10]
   N = 1 # (1000 ÷ len) * 2
   println("len = $len")
   x = zeros(len)
   print(" preallocate: ")
   @btime runn($x, $N, $len)
   print("      cached: ")
   @btime runn2($pool, $N, $len)
   print("garbage coll: ")
   @btime runn(zeros($len), $N, $len)
end

