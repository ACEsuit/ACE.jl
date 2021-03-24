using LinearAlgebra: norm, rank, svd, Diagonal

struct Orb
#	vals::Int
#   n::Int
	l::Int
	m::Int
end

# D^αβpq_ll,mm,kk
function coucoe(a::Orb, b::Orb, p::Orb, q::Orb, ll::T, mm::T, kk::T) where{T}
	if a.l == p.l && b.l==q.l
		return (-1)^(b.m-q.m) * CouCoe([ll a.l b.l], [mm a.m -b.m], [kk p.m -q.m])
#		return (-1)^(b.m-q.m) * CouCoe([ll a.l b.l], [kk p.m -q.m], [mm a.m -b.m])
	else
		return 0
	end
end


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
#	return val
end

# matrix D^αβpq_ll
function cou_coe_mat(a::Orb, b::Orb, p::Orb, q::Orb, ll::T) where {T}
   len = length(ll)
   CC = zeros(length(select_m(a,b,ll)), length(select_m(p,q,ll)))
#   println("???")
   if (length(select_m(a,b,ll)) == 1 || length(select_m(p,q,ll)) == 1)
	   #println("mm's value is",select_m(a,b,ll)[1])
	   CC[1,1] = coucoe(a ,b, p, q, ll, select_m(a,b,ll)[1], select_m(p,q,ll)[1])
   else
       for (im, m) in enumerate(select_m(a,b,ll)), (ik, k) in enumerate(select_m(p,q,ll))
           mm = zeros(len);kk=zeros(len);
	   	   for i = 1:len
		   	   mm[i] = m[i]
	   	   end
	   	   for i = 1:len
		   	   kk[i] = k[i]
	   	   end
	   	    mm = reshape(mm,1,len)
	   	    kk = reshape(kk,1,len)
#	   	println("reshape complete!")
	   	    mm = convert(Array{Int64}, mm)
	   	    kk = convert(Array{Int64}, kk)
#			@show kk
#			@show mm
#	   	println("convert complete!")
       	    CC[im, ik] = coucoe(a ,b, p, q, ll, mm, kk)
   	    end
    end
    return CC
end

# SVD to D^αβpq_ll
function rc_basis(a::Orb, b::Orb, p::Orb, q::Orb,ll::T) where {T}
	CC = cou_coe_mat(a, b, p, q, ll)
#	println(CC)
	svdC = svd(CC')
	rk = rank(Diagonal(svdC.S))
#	println(rk)
	return svdC.U[:, 1:rk]', select_m(p,q,ll)
#	return svdC.U[1:rk,:]
end

function select_m(a::Orb, b::Orb, ll::T) where {T}
	i = maximum(length(ll));t=1;
	A = CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll)))
	A_temp = CartesianIndex()
	for j=1:length(A)
		AA = zeros(length(ll))
		for m = 1:length(ll)
			AA[m]=A[j][m]
		end
#		println(AA)
		if sum(AA)!=b.m-a.m
#			println("deleted")
		else
			A_temp = A[j]
#			println("reserved")
			t=j+1
			break
		end
	end
#	println(t,length(A))
	for jj=t:length(A)
		AA = zeros(length(ll))
		for m = 1:length(ll)
			AA[m]=A[jj][m]
		end
#		println(AA)
		if sum(AA)!=b.m-a.m
#			println("deleted")
		else
			A_temp = [A_temp;A[jj]]
#			println("reserved")
		end
	end
#	println("Seclection done!")
	return A_temp
	#A = copy(A_temp)
end

#function coupling_coeffs(bb, rotc::Rot3DCoeffs, φ::SP)
function coupling_coeffs(ll, φ::SP)
	coupling_coeffs = zeros(Float64,1,3);
    # bb = [ b1, b2, b3, ...)
    # bi = (μ = ..., n = ..., l = ..., m = ...)
    #    (μ, n) -> n; only the l and m are used in the angular basis
    if length(bb) == 0
       return [1.0,], [bb,]
    end
    # convert to a format that the Rotations3D implementation can understand
    # this utility function splits the bb = (b1, b2 ...) with each
    # b1 = (μ = ..., n = ..., l = ..., m = ...) into
    #    l, and a new n = (μ, n)
#    ll, nn = _b2llnn(bb)

    # now we can call the coupling coefficient construiction!!
    U, Ms = rc_basis(ll)

    # but now we need to convert the m spec back to complete basis function
    # specifications
#    rpibs = [ _nnllmm2b(nn, ll, mm) for mm in Ms ]
	return U, Ms
#    return U, rpibs
end
