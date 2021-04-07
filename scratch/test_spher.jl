
# a basic test to  debug the spherical vector implementation
# -----------------------------------------------------------------

using ACE, StaticArrays, LinearAlgebra
mat(X::AbstractVector{<:StaticVector}) = hcat(X...)
rotc = ACE.Rotations3D.Rot3DCoeffs()

##
L = 1
Lmax = 3
U = fill( SA[0.0im, 0.0im, 0.0im], ((Lmax+1)^2, (Lmax+1)^2) )
for l = 1:2:Lmax, m = -l:l, μ = -l:l
   u = ACE.Rotations3D.local_cou_coe(rotc, SA[l], SA[m], SA[μ], L, 1)
   row = ACE.SphericalHarmonics.index_y(l, m)
   col = ACE.SphericalHarmonics.index_y(l, μ)
   U[row, col] = u
end

##

# check for  F(Qr) = D F(r)  ⇔   D' F(Qr) = F(r)

sh = ACE.SphericalHarmonics.SHBasis(Lmax)
r = @SVector randn(3); r /= norm(r)
Y = ACE.evaluate(sh, r)
UxY = U * Y

Q, D = ACE.Wigner.rand_QD(1)
Y_Q = ACE.evaluate(sh, Q * r)
DxUxY_D = Ref(D') .* (U * Y_Q)

DxUxY_D ≈ UxY


##

using ACE
using StaticArrays, Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, SymmetricBasis, NaiveTotalDegree, PIBasis
using ACE.Random: rand_rot, rand_refl
using ACE.Wigner

# construct the 1p-basis
D = ACE.NaiveTotalDegree()
maxdeg = 6
ord = 1
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D, pin = 0, pout = 0)
# generate a configuration
nX = 1
Xs = rand(EuclideanVectorState, B1p.bases[1], nX)
cfg = ACEConfig(Xs)

##

L = 1
ord = 1
φ = ACE.SphericalVector(L; T = ComplexF64)
pibasis = PIBasis(B1p, ord, 3; property = φ, isreal=false)

spec = ACE.get_spec(pibasis)
# specodd = filter(b -> isodd(b[1].l) && (b[1].n == 1), spec)
specf = [ (n=1, l=l, m=m) for l = 0:Lmax for m = -l:l ]
If = [ findall(isequal([b]), spec)[1] for b in specf ]

U = fill( SA[0.0im, 0.0im, 0.0im], length(specf), length(pibasis))
for (i, b) in enumerate(specf)
   for μ = -b.l:b.l
      bμ = (n = b.n, l = b.l, m = μ)
      iμ = findall(isequal([bμ]), spec)[1]
      u = ACE.Rotations3D.local_cou_coe(rotc, SA[b.l], SA[b.m], SA[μ], L, 1)
      U[i, iμ] = u
   end
end

##

Xs1  = Xs[1:1]
cfg1 = ACEConfig(Xs1)
AA = evaluate(pibasis, cfg1)
Y = evaluate(sh, Xs[1].rr)
 AA[If] / AA[If[1]] ≈ Y/Y[1]
##

Xs1  = Xs[1:1]
cfg1 = ACEConfig(Xs1)
AA = evaluate(pibasis, cfg1)
UxAA = U * AA

Q, D = ACE.Wigner.rand_QD(1)
Qcfg = ACEConfig( Ref(Q) .* Xs1 )
DtxUxAA_Q = Ref(D') .* (U * evaluate(pibasis, Qcfg))
DtxUxAA_Q ≈ UxAA

##

mat(DtxUxAA_Q - UxAA)[:,1]

##
