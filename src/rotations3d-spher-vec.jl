
# Index of entries in D matrix (sign included)
struct D_Index
	sign::Int64
	μ::Int64
	m::Int64
end

# Equation (1.1) - forms the covariant matrix D(Q)(indices only)
function Rotation_D_matrix(L::Integer)
	if L<0
		error("Orbital type shall be represented as a positive integer!")
	end
    D = Array{D_Index}(undef, 2 * L + 1, 2 * L + 1)
    for i = 1 : 2 * L + 1
        for j = 1 : 2 * L + 1
            D[j,i] = D_Index(1, i - 1 - L, j - 1 - L);
        end
    end
	return D
end

# Equation (1.1) - forms the covariant matrix D(Q)(indices only)
function Rotation_D_matrix_ast(L::Integer)
	if L<0
		error("Orbital type shall be represented as a positive integer!")
	end
    D = Array{D_Index}(undef, 2 * L + 1, 2 * L + 1)
    for i = 1 : 2 * L + 1
        for j = 1 : 2 * L + 1
            D[i,j] = D_Index((-1)^(i+j), -(i - 1 - L), -(j - 1 - L));
        end
    end
	return D
end


# Equation (1.2) - vector value coupling coefficients
# ∫_{SO3} D^{ll}_{μμmm} D^*(Q) e^t dQ -> 2L+1 column vector
function vec_cou_coe(rotc::Rot3DCoeffs{T},
					   ll::StaticVector{N},
	                   mm::StaticVector{N},
					   μμ::StaticVector{N},
					   L::Integer, t::Integer) where {T,N}
	if t > 2L + 1 || t < 0
		error("Rotation D matrix has no such column!")
	end
	Z = zeros(2L + 1)
	D = Rotation_D_matrix_ast(L)
	Dt = D[:,t]   # D^* ⋅ e^t
	μt = [Dt[i].μ for i in 1:2L+1]
	mt = [Dt[i].m for i in 1:2L+1]
	LL = [ll; L]
	for i = 1:(2L + 1)
		MM = [mm; mt[i]]
		KK = [μμ; μt[i]]
		Z[i] = Dt[i].sign * rotc(LL, MM, KK)
	end
	return Z
end

# Equation (1.5) - possible set of mm w.r.t. index ll & vector k
function collect_m(ll::StaticVector{N}, k::T) where {N,T}
	d = length(k)
	A = CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll)));
	B = Array{typeof(A[1].I)}(undef, 1, prod(size(A)))
	t = 0
	for i in A
		if prod(sum(i.I) .+ k) == 0
			t = t + 1
			B[t] = i.I
		end
	end
	B = [SVector(i) for i in B[1:t]]
	return B
end

function gramian_all(A::Rot3DCoeffs{T}, ll::StaticVector{N},
	                 L::Integer) where {T,N}
	LenM = 0
	D = Rotation_D_matrix_ast(L)
	D1 = D[:,1]
	μt = [D1[i].μ for i in 1:2L+1]
	μ_list = collect_m(ll,μt)
	Z = fill(zeros(2L + 1), (length(μ_list), length(μ_list)))
	for t = 1:2L+1
		Dt = D[:,t]
		mt = [Dt[i].m for i in 1:2L+1];
		m_list = collect_m(ll,mt)
		for (im, mm) in enumerate(m_list), (iμ, μμ) in enumerate(μ_list)
			Z[iμ, im + LenM] = vec_cou_coe(A, ll, mm, μμ, L, t)
		end
		LenM += length(m_list)
	end
	return Z' * Z, Z, μ_list
end

function rc_basis_all(A::Rot3DCoeffs{T}, ll::StaticVector{N},
	                  L::Integer) where {N,T}
	G, C, μ_list = gramian_all(A, ll, L)
	S = svd(G)
	rk = rank(G; rtol =  1e-8)
	Urcpi = fill(zeros(2L + 1), (rk, length(μ_list)))
	U = S.U[:, 1:rk]
	Sigma = S.S[1:rk]
	Urcpi = C * U * Diagonal(sqrt.(Sigma))^(-1)
	return Urcpi', μ_list
end

# Equation (1.12) - Gramian over nn
function Gramian(A::Rot3DCoeffs,
			     nn::StaticVector{N},
			     ll::StaticVector{N},
				 L::Integer) where {N}
	Uri, μ_list = rc_basis_all(A, ll, L)
	G = zeros(size(Uri)[1], size(Uri)[1])
	for σ in permutations(1:N)
       if (nn[σ] != nn) || (ll[σ] != ll); continue; end
       for (iU1, mm1) in enumerate(μ_list), (iU2, mm2) in enumerate(μ_list)
          if mm1[σ] == mm2
             for i1 = 1:size(Uri)[1]
				 for i2 = 1:size(Uri)[1]
                 	G[i1, i2] += Uri[i1, iU1] * Uri[i2, iU2]'
				end
             end
          end
       end
    end
    return G, Uri, μ_list
end

## Equation (1.13) - LI coefficients(& corresponding μ) over nn, ll
function yvec_symm_basis(A::Rot3DCoeffs,
				             nn::StaticVector{N},
								 ll::StaticVector{N},
								 L::Integer) where {N}
	G, C, μ_list= Gramian(A, nn, ll, L)
	S = svd(G)
	rk = rank(G; rtol =  1e-8)
	Urcpi = fill(zeros(2L + 1), (rk, length(μ_list)))
	U = S.U[:, 1:rk]
	Sigma = S.S[1:rk]
	Urcpi = C' * U * Diagonal(sqrt.(Sigma))^(-1)
	return Urcpi', μ_list
end
