
using ACE, LinearAlgebra
using ACE: evaluate

B = ACE.Utils.rpi_basis( species = [:X, :Al], N = 4,
                         pin = 0, pcut = 0)


Rs, Zs, z0 = ACE.Random.rand_nhd(10, B.pibasis.basis1p.J, :Al)
Zs[1] = 0

function fcut_env(r, z, rcut, zcut)
   return (r - rcut)^2 * (z - zcut)^2
end

function fcut_bond(r, rcut)
   return (r - rcut)^2
end

function envelope(Rs, rcut, rcut_env, zcut_env)
   fenv = fcut_bond(norm(Rs[1]))
   r̂ = Rs[1] / norm(Rs[1])
   o = Rs[1]/2
   for i = 2:length(Rs)
      R = Rs[i]
      z = dot(R - o, r̂)
      r = norm(R - z * r̂)
      fenv *= fcut_env(r, z, rcut, zcut)
   end
   return fenv
end

function eval_bond(B, Rs, Zs, z0)
   r̂ = Rs[1] / norm(Rs[1])
   o = Rs[1]/2
   Rr = map( r_ -> (r = r_ - o; o + r - 2*dot(r,r̂)*r̂), Rs )
   Rr[1] = Rs[1]
   return ( (evaluate(B, Rs, Zs, z0) + evaluate(B, Rr, Zs, z0))
            * envelope(Rs, rcut, rcut_env, zcut_env) )
end

b0 = eval_bond(B, Rs, Zs, z0)

# reversed configuration to test we have the right symmetry
Rs1 = [ r - Rs[1] for r in Rs ]
Rs1[1] = - Rs[1]
b1 = eval_bond(B, Rs1, Zs, z0)
b0 ≈ b1

B.pibasis.basis1p.J.
cutoff(B.pibasis.basis1p.J)
using Plots
rr = range(0, cutoff(B) + 1, length=100)
B1to5 = zeros(5, length(rr))
for (i, r) in enumerate(rr)
   B1to5[:, i] = evaluate(B.pibasis.basis1p.J, r)[1:5]
end
plot(rr, B1to5', ylims=[-2, 2])
vline!([B.pibasis.basis1p.J.rl, B.pibasis.basis1p.J.ru], lw=2, c=:black)
