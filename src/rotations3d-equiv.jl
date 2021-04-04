
"""
`Rot3DCoeffsEquiv: ` storing recursively precomputed coefficients for a
rotation-invariant basis.
"""

using LinearAlgebra: dot

struct Rot3DCoeffsEquiv{T,L}<: R3DC{T}
   vals::Vector{Dict}
   cg::ClebschGordan{T}
end


Rot3DCoeffsEquiv(φ::Invariant,T=Float64) = Rot3DCoeffsEquiv{T,0}(Dict[], ClebschGordan(T))

Rot3DCoeffsEquiv( φ::EuclideanVector,T=Float64) = Rot3DCoeffsEquiv{T,1}(Dict[], ClebschGordan(T))


struct MRangeEq{N, T2}
   ll::SVector{N, Int}
   cartrg::T2
   L::Int
end

Base.length(mr::MRangeEq) = sum(_->1, _mrange(mr.ll, mr.L))

_mrange(ll, L) = MRangeEq(ll, Iterators.Stateful(
               filter((x) -> abs(sum(x))<= L, Tuple.(CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll)))))
                     ),L)

function Base.iterate(mr::MRangeEq, args...)
	while true
		if isempty(mr.cartrg)
   		return nothing
		end
		mpre = popfirst!(mr.cartrg)
		return SVector(mpre), nothing
	end
end


function get0(L::Int,T=Float64)
	if L== 0
		return T(0)
	elseif L == 1
		return @SArray zeros(Complex{T},3,3)
		#SMatrix{3, 3, Complex{T}, 9}(0, 0, 0, 0, 0, 0, 0, 0, 0)
	else
		#For L== 2 @SArray zeros(Complex{T},3,3,3)
		ErrorException("Only types for L in {0,1} implemented")
	end
end

dicttype(A::Rot3DCoeffsEquiv, N::Integer) = dicttype(A::Rot3DCoeffsEquiv,Val(N))

dicttype(A::Rot3DCoeffsEquiv{T,L},::Val{N}) where {T,L,N} =
   Dict{Tuple{SVector{N,Int}, SVector{N,Int}, SVector{N,Int}}, typeof(get0(L,T))}

Rot3DCoeffsEquiv(L,T=Float64) = Rot3DCoeffsEquiv{T,L}(Dict[], ClebschGordan(T))

function get_vals(A::Rot3DCoeffsEquiv, valN::Val{N}) where {N}
	if length(A.vals) < N
		for n = length(A.vals)+1:N
			push!(A.vals, dicttype(A,n)())
		end
	end
   	return A.vals[N]::dicttype(A,valN)
end


function (A::Rot3DCoeffsEquiv{T,L})(ll::StaticVector{N},
							mm::StaticVector{N},
							kk::StaticVector{N}) where {T,L, N}
   if       abs(sum(mm)) > L ||
			abs(sum(kk)) > L ||
			!all(abs.(mm) .<= ll) ||
			!all(abs.(kk) .<= ll)
	  return get0(L,T)
   end
   vals = get_vals(A, Val(N))  # this should infer the type!
   key = _key(ll, mm, kk)
   if haskey(vals, key)
	  val  = vals[key]
   else
	  val = _compute_val(A, key...)
	  vals[key] = val
   end
   return val
end


function (A::Rot3DCoeffsEquiv{T,0})(ll::StaticVector{1},
							mm::StaticVector{1},
							kk::StaticVector{1}) where {T}
   if ll[1] == mm[1] == kk[1] == 0
	  return T(8 * pi^2)
   else
	  return get0(0,T)
   end
end

function (A::Rot3DCoeffsEquiv{T,1})(ll::StaticVector{1},
							mm::StaticVector{1},
							kk::StaticVector{1}) where {T}
   if ll[1] == 1 && abs(mm[1]) <= 1 && abs(kk[1]) <= 1
	  return  rmatrices[(mm[1],kk[1])]
   else
	  return get0(1,T)
   end
end

#function get0val(m,k,L)

#end
rmatrices = Dict(
  (-1,-1) => SMatrix{3, 3, ComplexF64, 9}(1/6, 1im/6, 0, -1im/6, 1/6, 0, 0, 0, 0),
  (-1,0) => SMatrix{3, 3, ComplexF64, 9}(0, 0, 0, 0, 0, 0, 1/(3*sqrt(2)), 1im/(3*sqrt(2)), 0),
  (-1,1) => SMatrix{3, 3, ComplexF64, 9}(-1/6, -1im/6, 0, -1im/6, 1/6, 0, 0, 0, 0),
  (0,-1) => SMatrix{3, 3, ComplexF64, 9}(0, 0, 1/(3*sqrt(2)), 0, 0, -1im/(3*sqrt(2)), 0, 0, 0),
  (0,0) => SMatrix{3, 3, ComplexF64, 9}(0, 0, 0, 0, 0, 0, 0, 0, 1/3),
  (0,1) => SMatrix{3, 3, ComplexF64, 9}(0, 0, -1/(3*sqrt(2)), 0, 0, -1im/(3*sqrt(2)), 0, 0, 0),
  (1,-1) => SMatrix{3, 3, ComplexF64, 9}(-1/6, 1im/6, 0, 1im/6, 1/6, 0, 0, 0, 0),
  (1,0) => SMatrix{3, 3, ComplexF64, 9}(0, 0, 0, 0, 0, 0, -1/(3*sqrt(2)), 1im/(3*sqrt(2)), 0),
  (1,1) => SMatrix{3, 3, ComplexF64, 9}(1/6, -1im/6, 0, 1im/6, 1/6, 0, 0, 0, 0)
  )

function (A::Rot3DCoeffsEquiv{T,L})(ll::StaticVector{1},
							 mm::StaticVector{1},
							 kk::StaticVector{1}) where {T,L}
	ErrorException("Not implemented for L = " + string(L))
end

function _compute_val(A::Rot3DCoeffsEquiv{T,L}, ll::StaticVector{N},
                                          mm::StaticVector{N},
                                          kk::StaticVector{N}) where {T,L,N}

     val = get0(L,T)
     llp = ll[1:N-2]
     mmp = mm[1:N-2]
     kkp = kk[1:N-2]
     for j = abs(ll[N-1]-ll[N]):(ll[N-1]+ll[N])
        if abs(kk[N-1]+kk[N]) > j || abs(mm[N-1]+mm[N]) > j
           continue
        end
  		cgk = try
  			A.cg(ll[N-1], kk[N-1], ll[N], kk[N], j, kk[N-1]+kk[N])
  		catch
  			@show (ll[N-1], kk[N-1], ll[N], kk[N], j, kk[N-1]+kk[N])
  			get0(L,T)
  		end
  		cgm = A.cg(ll[N-1], mm[N-1], ll[N], mm[N], j, mm[N-1]+mm[N])
  		if cgk * cgm  != 0
  			val += cgk * cgm * A( SVector(llp..., j),
  								       SVector(mmp..., mm[N-1]+mm[N]),
  								       SVector(kkp..., kk[N-1]+kk[N]) )
  		end
     end
     return val
 end

function vec3_symm_basis(A::Rot3DCoeffsEquiv{T,1},
						 nn::SVector{N, TN},
						 ll::SVector{N, Int}) where {T, N, TN}
	Ure, Mre = re_basis(A, ll)
	G = _gramian(nn, ll, Ure, Mre)
   	S = svd(G)
   	rk = rank(G; rtol =  1e-7)
	Urpe = S.U[:, 1:rk]'
	Utemp = Diagonal(sqrt.(S.S[1:rk])) * Urpe * Ure
	Utilde = zeros(SArray{Tuple{3},Complex{T},1,3}, rk, length(Mre))
	for α in 1:rk
		for (imu, mu) in enumerate(Mre)
			for k in 1:3
				for (i, mm) in enumerate(Mre)
					Utilde[α,imu] += Utemp[α,i] * A(ll,mu,mm)[:,k]
				end
			end
		end
	end

	return Utilde, Mre
end

function re_basis(A::Rot3DCoeffsEquiv{T}, ll::SVector; ordered=false) where {T}
	GGre, Mre = _gramianERot(A, ll)
	S = svd(GGre)
	rk = rank(Diagonal(S.S))
	return Diagonal(sqrt.(S.S[1:rk])) * S.U[:, 1:rk]', Mre
end

function _gramianERot(A::Rot3DCoeffsEquiv{T,1}, ll::SVector{N}) where{T,N}
	Mre = collect(_mrange(ll, 1))
  	len = length(Mre)*3
	GGre = zeros(Complex{T}, len, len)

	for (i1, (k1,mm1)) in enumerate(Iterators.product(1:3, Mre)), (i2, (k2,mm2)) in enumerate(Iterators.product(1:3, Mre))
		for mu1 in Mre, mu2 in Mre
			#show(typeof(dot(A(ll, mu2, mm2)[:,k2],A(ll, mu1, mm1)[:,k1])))
			GGre[i1, i2] += dot(A(ll, mu2, mm2)[:,k2],A(ll, mu1, mm1)[:,k1])
      	end
	end
   	return GGre, Mre
	#@assert all(abs.(imag(GG)) .<= imag_tol)
	#return real.(GG)
end
