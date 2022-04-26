

using ACE, BenchmarkTools

pool = ACE.ArrayCache{Float64}()
pool32 = ACE.ArrayCache{Float32}()
genpool = ACE.GenArrayCache()

##

function dosomething!(_A, x)
   A = parent(_A)
   @inbounds begin 
      A[1] = 1; A[2] = x 
      for n = 3:length(A)
         A[n] = 2 * x * A[n-1] - A[n-2]
      end
   end
   return _A 
end

function acquire_N(pool, N, len)
   x = rand() 
   for n = 1:N 
      x -= 1e-12
      A = ACE.acquire!(pool, Float64, len)
      dosomething!(A, x)
      ACE.release!(A)
   end
   return nothing
end

##

N = 1000
len = 100
acquire_N(pool, N, len)
acquire_N(genpool, N, len)
acquire_N(pool32, N, len)

@btime acquire_N($pool, $N, $len)
@btime acquire_N($genpool, $N, $len)
@btime acquire_N($pool32, $N, $len)

##

A = ACE.acquire!(pool, Float64, len)
gA = ACE.acquire!(genpool, Float64, len)
aA = ACE.acquire!(pool32, Float64, len)
x = rand()

@btime dosomething!($A, $x)
@btime dosomething!($gA, $x)
@btime dosomething!($aA, $x)

##

# @profview acquire_N(genpool, 100_000_000, len)

##
nothing
##

# ##

# function runn(x, N, len)
#    @assert length(x) >= len
#    fill!(x, 0)
#    for n = 1:N, i = 1:len 
#       x[i] += i/n
#    end 
#    return nothing 
# end

# x = rand(1000)
# @btime runn($x, 1000, 1000) 

# ##

# function runn2(cache, N, len)
#    x = ACE.acquire!(cache, len)
#    fill!(x.A, 0)
#    for n = 1:N, i = 1:len 
#       x[i] += i/n
#    end 
#    ACE.release!(x)
#    return nothing 
# end

# @btime runn2($pool, 1000, 1000) 

# ##

# N = 1000 

# for len in [1000, 300, 100, 30, 10]
#    N = 1 # (1000 ÷ len) * 2
#    println("len = $len")
#    x = zeros(len)
#    print(" preallocate: ")
#    @btime runn($x, $N, $len)
#    print("      cached: ")
#    @btime runn2($pool, $N, $len)
#    print("garbage coll: ")
#    @btime runn(zeros($len), $N, $len)
# end

