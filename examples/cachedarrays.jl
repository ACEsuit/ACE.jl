
using BenchmarkTools
using ACE

function dosomething!(A, x)
   @inbounds begin 
      A[1] = 1; A[2] = x 
      for n = 3:length(A)
         A[n] = 2 * x * A[n-1] - A[n-2]
      end
   end
   return A 
end

pool = ACE.

A1 = zeros(30)


po@btime dosomething!($A, 0.123)
@btime dosomething!($A1, 0.123)
@btime dosomething!($A2, 0.123)
