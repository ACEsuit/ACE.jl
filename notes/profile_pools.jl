
# This script shows why we use the Stack datastructure to implement th e
# objectpools. Using Vector as a stack is also fast but that caused 
# weird unexplainable bugs. Stack seems to work fine. 
#

using StaticArrays, BenchmarkTools, DataStructures


##

module ObjectPools

using Base.Threads: nthreads, threadid

struct VectorPool{T}
    #        tid    stack    object 
    arrays::Vector{Set{Vector{T}}}
    VectorPool{T}() where {T} = new( [ Set{Vector{T}}() for _=1:nthreads() ] )
end

function acquire!(pool::VectorPool{T}, len::Integer) where {T}
    tid = threadid()
    if !isempty(pool.arrays[tid]) > 0     
        x = pop!(pool.arrays[tid])
        if len != length(x)
            resize!(x, len)
        end
        return x 
    else
        return Vector{T}(undef, len)
    end
end

function release!(pool::VectorPool{T}, x::Vector{T}) where {T}
    tid = threadid() 
    push!(pool.arrays[tid], x)
    return nothing 
end

end

##

N = 1000 
T = SVector{3, Float64}
vecpool = Vector{T}[] 
setpool = Set{Vector{T}}()
stackpool = Stack{Vector{T}}()

for _=1:5
   push!(vecpool, rand(T, 1000))
   push!(setpool, rand(T, 1000))
   push!(stackpool, rand(T, 1000))
end

##

function runn(N, f, args...)
   for n = 1:N 
      f(args...)
   end 
end 

function poppush(pool)
   x = pop!(pool)  # 137 
   push!(pool, x)  # 199 
end

runn(3, poppush, vecpool)
runn(3, poppush, setpool)
runn(3, poppush, stackpool)

##

@btime poppush($vecpool)
@btime poppush($setpool)
@btime poppush($stackpool)


##

Profile.clear()
let pool = pool
   @profile runn(100_000_000, poppush, pool)
end

Profile.print()
