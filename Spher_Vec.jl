using StaticArrays
using LinearAlgebra: norm, rank, svd, Diagonal
using ACE, StaticArrays, ACE.SphericalHarmonics;
using ACE.SphericalHarmonics: index_y;
using ACE.Rotations3D
using ACE: evaluate
using Combinatorics: permutations


## Stucture Orbitaltype is nothing but the SphericalVector
#  I'd like to add some subset of SphericalVector so that
#  we could classify the blocks that we focus on...

abstract type Orbitaltype end

struct Orbt <: Orbitaltype
    val::Int64
end

# Todo: In fact, l index can be neglected!!
"""
`D_Index` forms the indices of each entry in rotation D matrix
We are not interested in the exact value but the indices only
"""
struct D_Index
	l::Int64
    μ::Int64
	m::Int64
end

## 1-D coupling coueffcient used in ACE
"""
`CoeCoe` function could be replaced by that in Rotation3D,
one difference is that I did not specify the type of indices,
but this can also be done in Rotation3D
"""

function CouCoe(ll, mm, kk)
   N = maximum(size(ll))
   if N == 1
   	if ll[1] == mm[1] == kk[1] == 0
      	return 1
   	else
      	return 0
   end
   elseif N == 2
   	if ll[1] != ll[2] || sum(mm) != 0 || sum(kk) != 0
      	return 0
   	else
      	return 8 * pi^2 / (2*ll[1]+1) * (-1)^(mm[1]-kk[1])
   	end
	else
		val = 0
		llp = ll[1:N-2]'
		mmp = mm[1:N-2]'
		kkp = kk[1:N-2]'
		for j = abs(ll[N-1]-ll[N]):(ll[N-1]+ll[N])
			if abs(kk[N-1]+kk[N]) > j || abs(mm[N-1]+mm[N]) > j
		   	continue
			end
	  		cgk = clebschgordan(ll[N-1], kk[N-1], ll[N], kk[N], j, kk[N-1]+kk[N])
	  		cgm = clebschgordan(ll[N-1], mm[N-1], ll[N], mm[N], j, mm[N-1]+mm[N])
	  		if cgk * cgm  != 0
		  		val += cgk * cgm * CouCoe([llp j], [mmp mm[N-1]+mm[N]], [kkp kk[N-1]+kk[N]])
	  		end
		end
		return val
	end
end

## The end of CC...

## Begin of my code

# Equation (1.1) - forms the covariant matrix D(Q)(indices only)
function Rotation_D_matrix(φ::Orbitaltype)
	if φ.val<0
		error("Orbital type shall be represented as a positive integer!")
	end
    D = Array{D_Index}(undef, 2 * φ.val + 1, 2 * φ.val + 1)
    for i = 1 : 2 * φ.val + 1
        for j = 1 : 2 * φ.val + 1
            D[i,j] = D_Index(φ.val, i - 1 - φ.val, j - 1 - φ.val);
        end
    end
	return D
end

# Equation (1.2) - vector value coupling coefficients
function local_cou_coe(ll::StaticVector{N}, mm::StaticVector{N},
					   kk::StaticVector{N}, φ::Orbitaltype, t::Int64) where {N}
	if t > 2φ.val + 1
		error("Rotation D matrix has no such column!")
	end
	Z = zeros(Complex{Float64},2φ.val + 1);
	D = Rotation_D_matrix(φ);
	Dt = D[:,t];
	μt = [Dt[i].μ for i in 1:2φ.val+1];
	mt = [Dt[i].m for i in 1:2φ.val+1];
	LL = [ll;φ.val];
	for i = 1 : 2φ.val + 1
		Z[i] = CouCoe(LL, [mm;mt[i]], [kk;μt[i]]);
	end
	return Z
end

# Equation (1.5) - possible set of mm w.r.t. vector k
function collect_m(ll::StaticVector{N}, k::T) where {N,T}
	d = length(k);
	A = CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll)));
	B = Array{typeof(A[1].I)}(undef, 1, prod(size(A)))
	t = 0;
	for i in A
		if prod(sum(i.I) .+ k) == 0
			t = t + 1;
			B[t] = i.I;
		end
	end
	B = [SVector(i) for i in B[1:t]]
	return B
end

# Equation(1.7) & (1.6) respectively - gramian
function gramian(ll::StaticVector{N}, φ::Orbitaltype, t::Int64) where{N}
	D = Rotation_D_matrix(φ);
	Dt = D[:,t];
	μt = [Dt[i].μ for i in 1:2φ.val+1];
	mt = [Dt[i].m for i in 1:2φ.val+1];
	m_list = collect_m(ll,mt);
	μ_list = collect_m(ll,μt);
	Z = [zeros(2φ.val + 1) for i = 1:length(μ_list), j = 1:length(m_list)];
	for (im, mm) in enumerate(m_list), (iμ, μμ) in enumerate(μ_list)
		Z[iμ,im] = local_cou_coe(ll, mm, μμ, φ, t);
	end
	return Z' * Z, Z;
end

# Equation (1.8) - LI set w.r.t. t & ll (not for nn for now)
function rc_basis(ll::StaticVector{N}, φ::Orbitaltype, t::Int64) where {N}
	G, C = gramian(ll, φ, t);
	D = Rotation_D_matrix(φ);
	Dt = D[:,t];
	μt = [Dt[i].μ for i in 1:2φ.val+1];
	mt = [Dt[i].m for i in 1:2φ.val+1];
	S = svd(G);
	rk = rank(G; rtol =  1e-8);
	μ_list = collect_m(ll,μt)
	Urcpi = [zeros(2φ.val + 1) for i = 1:rk, j = 1:length(μ_list)];
	U = S.U[:, 1:rk];
	Sigma = S.S[1:rk]
	Urcpi = C * U * Diagonal(sqrt.(Sigma))^(-1);
	return Urcpi', μ_list
end

# Equation (1.10) - Collecting all t and sorting them in order
function rcpi_basis_all(ll::StaticVector{N}, φ::Orbitaltype) where {N}
	Urcpi_all, μ_list = rc_basis(ll, φ, 1);
	if φ.val ≠ 0
		for t = 2 : 2φ.val+1
			Urcpi_all = [Urcpi_all; rc_basis(ll, φ, t)[1]];
		end
	end
	return Urcpi_all, μ_list
end

## From now on I will try to do the second round of SVD to obtain LI w.r.t. nn

# Equation (1.12) - Gramian over nn
function Gramian(nn::StaticVector{N}, ll::StaticVector{N}, φ::Orbitaltype) where {N}
	Uri, Mri = rcpi_basis_all(ll, φ);
#	m_list = collect_m(ll,mt)
	G = zeros(Complex{Float64}, size(Uri)[1], size(Uri)[1]);
	for σ in permutations(1:N)
       if (nn[σ] != nn) || (ll[σ] != ll); continue; end
       for (iU1, mm1) in enumerate(Mri), (iU2, mm2) in enumerate(Mri)
          if mm1[σ] == mm2
             for i1 = 1:size(Uri)[1]
				 for i2 = 1:size(Uri)[1]
                 	G[i1, i2] += Uri[i1, iU1] * Uri[i2, iU2]'
				end
             end
          end
       end
    end
    return G, Uri
end

"""
'Rcpi_basis_final' function is aiming to take the place of 'yvec_symm_basis'
in rotation3D.jl but still have some interface problem to be discussed
"""
# Equation (1.13) - LI coefficients(& corresponding μ) over nn, ll
function Rcpi_basis_final(nn::StaticVector{N}, ll::StaticVector{N}, φ::Orbitaltype) where {N}
	if mod(sum(ll) + φ.val, 2) ≠ 0
		if mod(sum(ll), 2) ≠ 0
			@warn ("To gain reflection covariant, sum of `ll` shall be even")
		else
			@warn ("To gain reflection covariant, sum of `ll` shall be odd")
		end
	end
	G, C = Gramian(nn, ll, φ);
	D = Rotation_D_matrix(φ);
	Dt = D[:,1];
	μt = [Dt[i].μ for i in 1:2φ.val+1];
#	mt = [Dt[i].m for i in 1:2φ.val+1];
	S = svd(G);
	rk = rank(G; rtol =  1e-8);
	μ_list = collect_m(ll,μt)
	Urcpi = [zeros(2φ.val + 1) for i = 1:rk, j = 1:length(μ_list)];
	U = S.U[:, 1:rk];
	Sigma = S.S[1:rk]
	Urcpi = C' * U * Diagonal(sqrt.(Sigma))^(-1);
	return Urcpi', μ_list
end
## End of LI of nn


## A test for ss, sp, sd blocks - with spherical harmonic only and all model parameters equal to 0

# Preliminary - PI basis without radial function
function PIbasis(ll::T, mm::T, R::SVector{N, Float64}) where{T,N}
    k = maximum(size(ll))
#    @show N
    A_part = 0;
    A = 1;
    for i = 1:k
        for j = 1:N/3
            Y = evaluate(SH, SVector(R.data[3*j-2:3*j]));
            A_part = A_part + Y[index_y(ll[i], mm[i])];
        end
        A = A * A_part;
        A_part = 0;
    end
    return A
end

# Preliminary - from 3D-rotation matrix $$Q$$ to rotation angle $$α, β, γ$$
function Mat2Ang(Q)
	return atan(Q[1,3],Q[3,3]), asin(-Q[2,3]), atan(Q[2,1],Q[2,2]);
end

function Wigner_D(μ,m,l,α,β,γ)
	return exp(-im*α*μ) * wigner_d(μ,m,l,β)  * exp(-im*γ*m)
end

function wigner_d(μ, m, l, β)
    fc1 = factorial(l+m)
    fc2 = factorial(l-m)
    fc3 = factorial(l+μ)
    fc4 = factorial(l-μ)
    fcm1 = sqrt(fc1 * fc2 * fc3 * fc4)

    cosb = cos(β / 2.0)
    sinb = sin(β / 2.0)

    p = m - μ
    low  = max(0,p)
    high = min(l+m,l-μ)

    temp = 0.0
    for s = low:high
       fc5 = factorial(s)
       fc6 = factorial(l+m-s)
       fc7 = factorial(l-μ-s)
       fc8 = factorial(s-p)
       fcm2 = fc5 * fc6 * fc7 * fc8
       pow1 = 2 * l - 2 * s + p
       pow2 = 2 * s - p
       temp += (-1)^(s+p) * cosb^pow1 * sinb^pow2 / fcm2
    end
    temp *= fcm1

    return temp
end

function rot_D(φ,Q)
	Mat_D = zeros(Complex{Float64}, 2φ.val + 1, 2φ.val + 1);
	D = Rotation_D_matrix(φ);
	α, β, γ = Mat2Ang(Q);
	for i = 1 : 2φ.val + 1
		for j = 1 : 2φ.val + 1
			#Mat_D[i,j] = (-1)^(i+j) * Wigner_D(D[i,j].μ, D[i,j].m, D[i,j].l, α, β, γ);
			Mat_D[i,j] = Wigner_D(D[i,j].μ, D[i,j].m, D[i,j].l, α, β, γ);
		end
	end
	return Mat_D
end

function rot_D(φ,α::Float64,β::Float64,γ::Float64)
	Mat_D = zeros(Complex{Float64}, 2φ.val + 1, 2φ.val + 1);
	D = Rotation_D_matrix(φ);
	for i = 1 : 2φ.val + 1
		for j = 1 : 2φ.val + 1
			#Mat_D[i,j] = (-1)^(i+j) * Wigner_D(D[i,j].μ, D[i,j].m, D[i,j].l, α, β, γ);
			Mat_D[i,j] = Wigner_D(D[i,j].μ, D[i,j].m, D[i,j].l, α, β, γ);
		end
	end
	return Mat_D
end

# Preliminary - Rotate R w.r.t. specific Q
function Rot(R::SVector{N, Float64},Q) where {N}
    RotR = []; RotTemp = []; ii = 1;
#    K = randn(3, 3);
#    K = K - K';
#    Q = SMatrix{3,3}(rand([-1,1]) * exp(K)...);
    RotR = Q*R[3*ii-2:3*ii];
    if N/3 > 1
        for ii = 2:N/3
            RotTemp = SVector(R.data[3*ii-2:3*ii]);
            RotR = [RotR; Q*RotTemp];
        end
    end
    RotR = SVector(RotR)
    return RotR
end

# Check the correctness of Mat2Ang
function rotz(α)
	return [cos(α) sin(α) 0; -sin(α) cos(α) 0; 0 0 1];
end

function rotx(α)
	return [1 0 0; 0 cos(α) sin(α); 0 -sin(α) cos(α)];
end

function Ang2Mat(α,β,γ)
	return rotz(γ)*rotx(β)*rotz(α);
end

function test_M2A(Q)
	α, β, γ = Mat2Ang(Q);
	return Ang2Mat(α,β,γ)
end

# Begin of test
function test(nn::StaticVector{T}, ll::StaticVector{T}, φ::Orbitaltype, R::SVector{N, Float64}) where{T,N}
	Z = zeros(Complex{Float64}, 2φ.val+1, 1);
#	U, μ_list = rcpi_basis_all(ll, φ);
	U, μ_list = Rcpi_basis_final(nn, ll, φ);
	UU = sum(U, dims = 1)
	Num_μ = length(UU);
	for i = 1: Num_μ
		Z = UU[i]' * PIbasis(ll, μ_list[i], R) + Z;
	end
	reshape(Z,2φ.val+1,1)
	return Z, svd(Z).S
end
## End of the test
