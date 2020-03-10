

@testset "OneOrthogonal" begin

##
@info("--------- Testing OneOrthogonal ----------")
using LinearAlgebra, Test
using SHIPs
using SHIPs.OrthPolys: OrthPolyBasis, OneOrthogonal
using JuLIP: evaluate, evaluate_d, evaluate!, evaluate_d!
using JuLIP: evaluate
##
# Assemble basis
N = 10  # degree
Nquad = 1000
dt =  2 / Nquad
tdf = range(-1.0+dt/2, 1.0-dt/2, length=Nquad)
ww = ones(Nquad) / Nquad
P = OrthPolyBasis(N,  0, 1.0, 2, -1.0, tdf, ww)
Q = OneOrthogonal(P)

# Evaluate basis
p = Matrix{Float64}(undef,Nquad,N)
q = Matrix{Float64}(undef,Nquad,N)
for i = 1:Nquad
    evaluate!(@view(p[i,:]), nothing, P, tdf[i])
    evaluate!(@view(q[i,:]), nothing, Q, tdf[i])
end

@info("Test spanning property")
@test rank(p'*q) == N

@info("Test orthogonality properties")
@test q'*Diagonal(ww)*q ≈ I
@test norm(q[:,1:end-1]'*Diagonal(ww)*one.(tdf)) < sqrt(eps())

@info("Test derivative")
ε = sqrt(eps())
q1 = evaluate(Q, -ε/2)
q2 = evaluate(Q, ε/2)
dq = evaluate_d(Q,0.0)
@test (q2.-q1)./ε ≈ dq


##

@info("Generate many-body 1-orth basis")

deg = 10  # degree
r0, rcut = 0.5, 2.5
trans = PolyTransform(2, r0)
t0, tcut = trans(r0), trans(rcut)
tdf = range(t0, tcut, length=1000)
ww = ones(Nquad) / Nquad
J = OrthPolyBasis(deg,  0, trans(r0), 2, trans(rcut), tdf, ww)
J1 = OneOrthogonal(J)
P = SHIPs.TransformedPolys(J1, trans, r0, rcut)

# SHIPBasis(spec::BasisSpec, J; T=Float64, kwargs...)
spec = SparseSHIP(2, deg; wL = 1.5,
                  filterfcn = ν -> SHIPs.OrthPolys.filter_oneorth(ν, J1))
shpB = SHIPBasis(spec, P)
pureB = SHIPs.PureBasis(shpB)

I2 = findall(shpB.aalists[1].len .== 1)
B2 = SHIPs.Utils.SubBasis(shpB, I2)

##


Rs = SHIPs.rand_vec(shpB.J, 3)
Zs = zeros(Int16, 5)
b = evaluate(shpB, Rs, Zs, 0)
bp = pureB(Rs)
@assert length(b) == length(shpB) == length(bp)

G, A_dot_1 = let nsamples = 10_000
   G = zeros(length(shpB), length(shpB))
   A_dot_1 = zeros(length(I2))
   for _ = 1:nsamples
      Rs = SHIPs.rand_vec(shpB.J, 2)
      b = pureB(Rs)
      G += b * b' / nsamples
      A_dot_1 += b[I2] / nsamples
   end
   G, A_dot_1
end

##
D = Diagonal( diag(G).^(-0.5) )
Gsc = D * G * D
@show cond(Gsc[1:9,1:9])  # this shows the A-basis is orthogonal
@show cond(Gsc)           # but not the ∏A basis
@show norm(A_dot_1, Inf)

end
