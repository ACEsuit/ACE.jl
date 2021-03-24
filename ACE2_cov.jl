using StaticArrays
using LinearAlgebra: norm, rank, svd, Diagonal

abstract type Orbitaltype end

struct Orbt <: Orbitaltype
    val::Int64
end

struct D_Index
	l::Int64
    μ::Int64
	m::Int64
end

# Equation (1.1)
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

# Equation (1.5)
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

# Equation (1.6)
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

# Red part of Equation (1.7)
function vec_cou_coe(ll::StaticVector{N}, mm::StaticVector{N}, φ::Orbitaltype, t::Int64) where{N}
	U = [@SVector zeros(2φ.val + 1)];
	U = SVector(U...);
	D = Rotation_D_matrix(φ);
	Dt = D[:,t];
	μt = [Dt[i].μ for i in 1:2φ.val+1];
	for (iμ1, μ1) in enumerate(collect_m(ll,μt))
		U += local_cou_coe(ll, mm, μ1, φ, t);
	end
	return U
end

# Equation (1.7)
function Gramian(ll::StaticVector{N}, φ::Orbitaltype, t::Int64) where {N}
	D = Rotation_D_matrix(φ);
	Dt = D[:,t];
	μt = [Dt[i].μ for i in 1:2φ.val+1];
	mt = [Dt[i].m for i in 1:2φ.val+1];
	m_list = collect_m(ll,mt)
	G = zeros(Complex{Float64}, length(m_list), length(m_list));
	for (im1, m1) in enumerate(m_list), (im2, m2) in enumerate(m_list)
		G[im1,im2] = vec_cou_coe(ll,m1,φ,t)' * vec_cou_coe(ll,m2,φ,t);
	end
	return G
end

##''' Alternatively, we could compute the "small gramian" '''

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

# rcpi_basis w.r.t. small gramian matrix?

function rcpi_basis(ll::StaticVector{N}, φ::Orbitaltype, t::Int64) where {N}
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

# Collecting all t and sort them in order
function rcpi_basis_all(ll::StaticVector{N}, φ::Orbitaltype) where {N}
	Urcpi_all, μ_list = rcpi_basis(ll, φ, 1);
	if φ.val ≠ 0
		for t = 2 : 2φ.val+1
			Urcpi_all = [Urcpi_all; rcpi_basis(ll, φ, t)[1]];
		end
	end
	return Urcpi_all, μ_list
end
## The end of small gramian test

# Equation (5.13)
function Rcpi_basis(ll::StaticVector{N}, φ::Orbitaltype, t::Int64) where {N}
	G = Gramian(ll, φ, t);
	D = Rotation_D_matrix(φ);
	Dt = D[:,t];
	μt = [Dt[i].μ for i in 1:2φ.val+1];
	mt = [Dt[i].m for i in 1:2φ.val+1];
	S = svd(G);
	rk = rank(G; rtol =  1e-8);
	μ_list = collect_m(ll,μt)
	Urcpi = [zeros(2φ.val + 1) for i = 1:rk, j = 1:length(μ_list)];
	U = S.U[:, 1:rk]';
	Sigma = S.S[1:rk]
	for i = 1:rk
		for (iμ, μ) in enumerate(μ_list)
#			for (im1, m) in enumerate(collect_m(ll,mt))
				Urcpi[i,iμ] = Sigma[i]^(-1) * U[i, im1] * local_cou_coe(ll, m, μ, φ, t)
#			end
		end
	end
	return Urcpi, μ_list
end

function Rcpi_basis_all(ll::StaticVector{N}, φ::Orbitaltype) where {N}
	Urcpi_all, μ_list = Rcpi_basis(ll, φ, 1);
#	if φ.val ≠ 0
#		for t = 2 : 2φ.val+1
#			Urcpi_all = [Urcpi_all; rcpi_basis(ll, φ, t)[1]];
#		end
#	end
	return Urcpi_all, μ_list
end


## A test for ss, sp, sd blocks - with spherical harmonic only and all model parameters equal to 0
function test(ll::StaticVector{T}, φ::Orbitaltype, R::SVector{N, Float64}) where{T,N}
	Z = zeros(Complex{Float64}, 2φ.val+1, 1);
	U, μ_list = rcpi_basis_all(ll, φ);
	UU = sum(U, dims = 1)
	Num_μ = length(UU);
	for i = 1: Num_μ
		Z = UU[i]' * PIbasis(ll, μ_list[i], R) + Z;
	end
	return Z
end
## End of the test
