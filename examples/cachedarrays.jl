
using BenchmarkTools

struct MyArray1{N, T}
   A::Array{T, N}
end

Base.length(mA::MyArray1) = length(mA.A)
Base.getindex(mA::MyArray1, I...) = getindex(mA.A, I...)
Base.setindex!(mA::MyArray1, I...) = setindex!(mA.A, I...)

struct MyArray2{N, T}
   A::Array{T, N}
end

Base.length(mA::MyArray2) = length(mA.A)

@propagate_inbounds function Base.getindex(mA::MyArray2, I...) 
   @boundscheck checkbounds(A, I...)
   @inbounds mA.A[I...]
end 

@propagate_inbounds function Base.setindex!(mA::MyArray2, val, I...) 
   @boundscheck checkbounds(A, I...)
   mA.A[I...] = val 
end


function dosomething!(A, x)
   @inbounds begin 
      A[1] = 1; A[2] = x 
      for n = 3:length(A)
         A[n] = 2 * x * A[n-1] - A[n-2]
      end
   end
   return A 
end

A = zeros(20)
A1 = MyArray1(A)
A2 = MyArray2(A)

@btime dosomething!($A, 0.123)
@btime dosomething!($A1, 0.123)
@btime dosomething!($A2, 0.123)
