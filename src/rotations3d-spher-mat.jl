

# ∫_{SO3} D^{ll}_{μμmm} D_{L1}^*(Q) e^{ab} D_{L2}(Q) dQ
# -> (2L_1+1)×(2L_2+1) matrix
# Has been checked to consist with `vec_cou_coe` when L2 = 0; b = 1;
function mat_cou_coe(rotc::Rot3DCoeffs{T},
					   ll::StaticVector{N},
	                   mm::StaticVector{N},
					   μμ::StaticVector{N},
					   L1::Integer, L2::Integer,
					   a::Integer, b::Integer) where {T,N}
	if a > 2L1 + 1 || a <= 0 || b > 2L2 +1 || b <= 0
		error("Rotation D matrices has no such element!")
	end
	Z = zeros(2 * L1 + 1, 2 * L2 + 1)
	Dp = rotation_D_matrix_ast(L1)
	Dq = rotation_D_matrix(L2)
	Dpa = Dp[:,a]
	Dqb = Dq[b,:]
	μa = [Dpa[i].μ for i in 1:2L1+1]
	ma = [Dpa[i].m for i in 1:2L1+1]
	μb = [Dqb[i].μ for i in 1:2L2+1]
	mb = [Dqb[i].m for i in 1:2L2+1]
	LL = [ll; L1; L2]
	for i = 1:(2 * L1 + 1)
		for j = 1:(2 * L2 + 1)
			MM = [mm; ma[i]; mb[j]]
			KK = [μμ; μa[i]; μb[j]]
			Z[i,j] = Dpa[i].sign * Dqb[j].sign * rotc(LL, MM, KK)
		end
	end
	return Z
end

# Has been checked to consist with vector case by choosing L2 = 0
function gramian(A::Rot3DCoeffs{T}, ll::StaticVector{N},
	                 L1::Integer, L2::Integer) where {T,N}
	LenM = 0
	Tempm = zeros(2 * L1 + 1,2 * L2 + 1)
	Dp = rotation_D_matrix_ast(L1)
	Dq = rotation_D_matrix(L2)
	Tempμ = SVector([i for i in -(L1+L2):(L1+L2)]...)
	μ_list = collect_m(ll,Tempμ)

	## TODO: the upper bound of #m_list(`(2L1 + 1)*length(μ_list)`) seems to be
	## too large and can be further reduced; just keep it for now...
	Z = fill(zeros(2L1 + 1, 2L2 + 1), (length(μ_list), (2L1 + 1)*length(μ_list)))

	for a = 1:2 * L1+1
		for b = 1:2 * L2+1
			for i = 1:2 * L1 + 1
				for j = 1: 2 * L2 +1
					Dpa = Dp[:,a]
					Dqb = Dq[b,:]
					ma = [Dpa[k].m for k in 1:2L1+1]
					mb = [Dqb[k].m for k in 1:2L2+1]
					Tempm[i,j] = ma[i]+mb[j]
				end
			end
			m_list = collect_m(ll,Tempm)
			for (im, mm) in enumerate(m_list), (iμ, μμ) in enumerate(μ_list)
				Z[iμ, im + LenM] = mat_cou_coe(A, ll, mm, μμ, L1, L2, a, b)
			end
			LenM += length(m_list)
		end
	end
	Z = Z[:, 1:LenM]

	gram = zeros(LenM, LenM)
	for i = 1:LenM
		for j = 1:LenM
			for t = 1:length(μ_list)
				gram[i,j] += tr(Z[t,i]' * Z[t,j])
			end
		end
	end

	return gram, Z, μ_list
end

# Has been checked to consist with vector case by choosing L2 = 0
function rc_basis_all(A::Rot3DCoeffs{T}, ll::StaticVector{N},
	                  L1::Integer, L2::Integer) where {N,T}
	G, C, μ_list = gramian(A, ll, L1, L2)
	S = svd(G)
	rk = rank(G; rtol =  1e-8)
	Urcpi = fill(zeros(2L1 + 1, 2L2 + 1), (rk, length(μ_list)))
	U = S.U[:, 1:rk]
	Sigma = S.S[1:rk]
	Urcpi = C * U * Diagonal(sqrt.(Sigma))^(-1)
	return Urcpi', μ_list
end

# Gramian over nn
function Gramian(A::Rot3DCoeffs,
			     nn::StaticVector{N},
			     ll::StaticVector{N},
				 L1::Integer, L2::Integer) where {N}
	Uri, μ_list = rc_basis_all(A, ll, L1, L2)
	G = zeros(size(Uri)[1], size(Uri)[1])
	for σ in permutations(1:N)
       if (nn[σ] != nn) || (ll[σ] != ll); continue; end
       for (iU1, mm1) in enumerate(μ_list), (iU2, mm2) in enumerate(μ_list)
          if mm1[σ] == mm2
             for i1 = 1:size(Uri)[1]
				 for i2 = 1:size(Uri)[1]
                 	G[i1, i2] += tr(Uri[i1, iU1] * Uri[i2, iU2]')
				end
             end
          end
       end
    end
    return G, Uri, μ_list
end

## LI matrix-valued coupling coefficients(& corresponding μ) over nn, ll
function mat_symm_basis(A::Rot3DCoeffs,
				        nn::StaticVector{N},
					    ll::StaticVector{N},
						L1::Integer, L2::Integer) where {N}
	G, C, μ_list= Gramian(A, nn, ll, L1, L2)
	S = svd(G)
	rk = rank(G; rtol =  1e-8)
	Urcpi = fill(zeros(2L1 + 1, 2L2 + 1), (rk, length(μ_list)))
	U = S.U[:, 1:rk]
	Sigma = S.S[1:rk]
	Urcpi = C' * U * Diagonal(sqrt.(Sigma))
	Z = fill(zeros(2L1 + 1, 2L2 + 1), (rk, length(μ_list)))
	for i = 1:rk
		for j = 1:length(μ_list)
			Z[i,j] = Urcpi[j,i];
		end
	end
	return Z, μ_list
end
