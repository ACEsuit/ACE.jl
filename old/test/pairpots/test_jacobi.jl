
@testset "Jacobi" begin

using SHIPs, Test

using SHIPs.JacobiPolys: Jacobi
using JuLIP: evaluate, evaluate_d
using JuLIP.Testing: print_tf

# copy-pasted from
# https://github.com/pjabardo/Jacobi.jl/blob/master/src/jac_poly.jl
# commit 9642d8f060203ddddeb17484a3aa8232022c7ba5

function jacobi(x, n, a, b)
   ox = one(x)
   zx = zero(x)
   if n==0
     return ox
   elseif n==1
     return ox/2 * (a - b + (a + b + 2)*x)
   end

   p0 = ox
   p1 = ox/2 * (a - b + (a + b + 2)*x)
   p2 = zx;

   for i = 1:(n-1)
      a1 = 2*(i+1)*(i+a+b+1)*(2*i+a+b);
      a2 = (2*i+a+b+1)*(a*a-b*b);
      a3 = (2*i+a+b)*(2*i+a+b+1)*(2*i+a+b+2);
      a4 = 2*(i+a)*(i+b)*(2*i+a+b+2);
      p2 = ox/a1*( (a2 + a3*x)*p1 - a4*p0);

      p0 = p1
      p1 = p2
   end

   return p2
end

# a standard recursion formula for the derivative of Jacobi Polynomials
djacobi(x, n, a, b) =  one(x)/2 * (a + b + n + 1) * jacobi(x, n-1, a+1, b+1)


@info("--------------  JACOBI TESTS ---------------")
@info("testing jacobi implementation against reference.")
for ntest = 1:30
   x = 2*rand() - 1
   α, β = rand(), rand()
   N = 30
   J = Jacobi(α, β, N, normalise=false)
   P = evaluate(J, x)
   dP = evaluate_d(J, x)
   Ptest = [ jacobi(x, n, α, β) for n = 0:N ]
   dPtest = [ djacobi(x, n, α, β) for n = 0:N ]
   print_tf((@test P ≈ Ptest))
   print_tf((@test dP ≈ dPtest))
end
println()

end
