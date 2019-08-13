
using Lasso
using GLM
using LinearAlgebra

##

Ψ = rand(100_000, 200)
Y = rand(100_000)
qrΨ = qr(Ψ)
Q = Matrix(qrΨ.Q)
η = norm(Y - Q * (Q' * Y)) / norm(Y)

R = Matrix(qrΨ.R)
y = Q' * Y
η1 = norm(y) / norm(Y)

target = sqrt(1.2^2 - 1) * η

c = nothing

for n = 1:length(p.λ)
   ηn = norm(R * p.coefs[:,n] - y) / norm(Y)
   @show ηn
   if ηn < target
      global c = p.coefs[:,n]
      break
   end
end

@show length(c.nzind)
@show norm(Ψ * c - Y)/norm(Y)
@show 1.2 * η

## ------------------------------------------------------------------------
