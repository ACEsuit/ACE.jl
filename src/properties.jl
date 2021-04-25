
import Base: -, +, *, filter
import LinearAlgebra: norm



@inline +(φ1::T, φ2::T) where {T <: AbstractProperty} = T( φ1.val + φ2.val )
@inline -(φ1::T, φ2::T) where {T <: AbstractProperty} = T( φ1.val - φ2.val )
@inline -(φ::T) where {T <: AbstractProperty} = T( -φ.val)
@inline *(a::Union{Number, AbstractMatrix}, φ::T) where {T <: AbstractProperty} =
      T(a * φ.val)
@inline *(φ::T, a::Union{Number, AbstractMatrix}) where {T <: AbstractProperty} =
      T(φ.val * a)
@inline norm(φ::AbstractProperty) = norm(φ.val)
@inline Base.length(φ::AbstractProperty) = length(φ.val)
@inline Base.size(φ::AbstractProperty) = size(φ.val)
@inline Base.zero(φ::T) where {T <: AbstractProperty} = T(zero(φ.val))
@inline Base.zero(::Type{T}) where {T <: AbstractProperty} = zero(T())

Base.isapprox(φ1::T, φ2::T) where {T <: AbstractProperty} =
      isapprox(φ1.val, φ2.val)

"""
`struct Invariant{D}` : specifies that the output of an ACE is
an invariant scalar.
"""
struct Invariant{T} <: AbstractProperty
   val::T
end

Invariant{T}() where {T <: Number} = Invariant{T}(zero(T))

Invariant(T::DataType = Float64) = Invariant{T}()

filter(φ::Invariant, b::Array) = ( length(b) <= 1 ? true :
     iseven(sum(bi.l for bi in b)) && iszero(sum(bi.m for bi in b))  )

rot3Dcoeffs(::Invariant, T=Float64) = Rot3DCoeffs(T)

# TODO: this is a naive implementation of differentiation.
#       cf https://github.com/ACEsuit/ACE.jl/issues/27
#       for further discussion
*(φ::Invariant, dAA::SVector) = φ.val * dAA

coco_init(::Invariant, l, m, μ, T, A) = (
      l == m == μ == 0 ? Invariant(T(1)) : Invariant(T(0))  )

coco_zeros(::Invariant, ll, mm, kk, T, A) = Invariant(zero(T))

coco_filter(::Invariant, ll, mm) =
            iseven(sum(ll)) && (sum(mm) == 0)

coco_filter(::Invariant, ll, mm, kk) =
            iseven(sum(ll)) && (sum(mm) == sum(kk) == 0)

coco_dot(u1::Invariant, u2::Invariant) = u1.val * u2.val

# --------------------- EuclideanVector

@doc raw"""
`struct EuclideanVector{D, T}` : specifies that the output $\varphi$ of an
ACE is an equivariant $\varphi \in \mathbb{R}^{3}$, i.e., it transforms under
$O(3)$ as
```math
      \varphi \circ Q = Q \cdot \varphi,
```
where $\cdot$ denotes the standard matrix-vector product.
"""
struct EuclideanVector{T} <: AbstractProperty
   val::SVector{3, T}
end

EuclideanVector{T}() where {T <: Number} = EuclideanVector{T}(zero(SVector{3, T}))

EuclideanVector(T::DataType=Float64) = EuclideanVector{T}()

filter(φ::EuclideanVector, b::Array) = ( length(b) <= 1 ? true :
             isodd( sum(bi.l for bi in b)) &&
            (abs(sum(bi.m for bi in b)) <= 1) )

rot3Dcoeffs(::EuclideanVector,T=Float64) = Rot3DCoeffsEquiv{T,1}(Dict[], ClebschGordan(T))

# differentiation - cf #27
*(φ::EuclideanVector, dAA::SVector) = φ.val * dAA'

coco_init(phi::EuclideanVector, l, m, μ, T, A) = (
      (l == 1 && abs(m) <= 1 && abs(μ) <= 1)
         ? [EuclideanVector{Complex{T}}(rmatrices[m,μ][:,k]) for k=1:3]
         : [EuclideanVector{Complex{T}}() for k=1:3]  )

coco_zeros(::EuclideanVector, ll, mm, kk, T, A) = [EuclideanVector{Complex{T}}() for k=1:3]

coco_filter(::EuclideanVector, ll, mm) =
            isodd(sum(ll)) && (abs(sum(mm)) <= 1)

coco_filter(::EuclideanVector, ll, mm, kk) =
      abs(sum(mm)) <= 1 &&
      abs(sum(kk)) <= 1 &&
      isodd(sum(ll))

coco_dot(u1::EuclideanVector, u2::EuclideanVector) = dot(u1.val,u2.val)

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


# --------------------- SphericalVector

struct SphericalVector{L, LEN, T} <: AbstractProperty
   val::SVector{LEN, T}
   _valL::Val{L}
end

# differentiation - cf #27
*(φ::SphericalVector, dAA::SVector) = φ.val * dAA'


getL(φ::SphericalVector{L}) where {L} = L

# L = 0 -> (0,0)
# L = 1 -> (0,0), (1,-1), (1,0), (1,1)  -> 4
# L = 3 ->  ... + 5 -> 9
# 1 + 3 + 5 + ... + 2*L+1
# = L + 2 * (1 + ... + L) = L+1 + 2 * L * (L+1) / 2 = (L+1)^2
function SphericalVector(L::Integer; T = Float64)
   LEN = 2L+1   # length of SH basis up to L
   return SphericalVector( zero(SVector{LEN, T}), Val{L}() )
end

function SphericalVector{L, LEN, T}(x::AbstractArray) where {L, LEN, T}
   @assert length(x) == LEN
   SphericalVector{L, LEN, T}( SVector{LEN, T}(x...), Val(L) )
end

SphericalVector{L, LEN, T}()  where {L, LEN, T} =
      SphericalVector( zero(SVector{LEN, T}), Val{L}() )

filter(φ::SphericalVector, b::Array) = ( length(b) <= 1 ? true :
        ( ( iseven(sum(bi.l for bi in b)) == iseven(getL(φ)) ) &&
         ( abs(sum(bi.m for bi in b)) <= getL(φ) )  ) )

rot3Dcoeffs(::SphericalVector, T::DataType=Float64) = Rot3DCoeffs(T)


const __rotcoeff_inv = Rotations3D.Rot3DCoeffs(Invariant())

using ACE.Wigner: rotation_D_matrix_ast, rotation_D_matrix

# Equation (1.2) - vector value coupling coefficients
# ∫_{SO3} D^{ll}_{μμmm} D^*(Q) e^t dQ -> 2L+1 column vector
function vec_cou_coe(rotc::Rot3DCoeffs{T},
					   ll::StaticVector{N},
	                   mm::StaticVector{N},
					   μμ::StaticVector{N},
					   L::Integer, t::Integer) where {T,N}
	if t > 2L + 1 || t <= 0
		error("Rotation D matrix has no such column!")
	end
	Z = zeros(2L + 1)
	D = rotation_D_matrix_ast(L)
	Dt = D[:,t]   # D^* ⋅ e^t
	μt = [Dt[i].μ for i in 1:2L+1]
	mt = [Dt[i].m for i in 1:2L+1]
	LL = [ll; L]
	for i = 1:(2L + 1)
		MM = [μμ; mt[i]]
		KK = [mm; μt[i]]
		Z[i] = Dt[i].sign * rotc(LL, MM, KK).val
	end
	return SphericalVector{L, 2L+1, Complex{T}}(Z)
end

function _select_t(φ::SphericalVector{L}, ll, mm, kk) where {L}
	M = sum(mm)
	K = sum(kk)
	D = rotation_D_matrix_ast(L)
	num_t = 0
	list_t = []
	for t = 1:2L+1
		Dt = D[:,t]
		μt = [Dt[i].μ for i in 1:2L+1]
		mt = [Dt[i].m for i in 1:2L+1]
		if prod(μt.+M)==0 && prod(mt.+K)==0
			list_t = [list_t;t]
			num_t = num_t+1
		end
	end
	if list_t == []
		return false, 0
	else
		return list_t, num_t
	end
end


coco_zeros(φ::TP, ll, mm, kk, T, A)  where {TP <: SphericalVector} =
		zeros(TP, _select_t(φ,ll,mm,kk)[2])

coco_dot(u1::SphericalVector, u2::SphericalVector) =
		dot(u1.val, u2.val)

coco_filter(φ::SphericalVector{L}, ll, mm) where {L} =
		iseven(sum(ll) + L) && (abs(sum(mm)) <=  L)

coco_filter(φ::SphericalVector{L}, ll, mm, kk) where {L} =
      iseven(sum(ll) + L) && (abs(sum(mm)) <=  L) && (abs(sum(kk)) <= L)


# coco_init(φ::SphericalVector{L}, l, m, μ, T, A) where {L} =
# 			[ vec_cou_coe(__rotcoeff_inv,
# 				 					SVector(l), SVector(m), SVector(μ), L, t)
# 				for t = 1:2*L+1 ]

function coco_init(φ::SphericalVector{L}, l, m, μ, T, A) where {L}
	list, num = _select_t(φ,SVector(l),SVector(m),SVector(μ))
	if num == 1
		t = list[1]
		return [vec_cou_coe(__rotcoeff_inv, SVector(l), SVector(m), SVector(μ), L, t)]
	else
		@warn("IS IT POSSIBLE???")
		return coco_zeros(φ, l, m, μ, T, A)
	end
end

# function coco_init(φ::SphericalVector{L}, l, m, μ, T, A) where {L}
# 	for t = 1:2*L+1
# 		Temp = vec_cou_coe(__rotcoeff_inv, SVector(l), SVector(m), SVector(μ), L, t)
# 		if !(norm(Temp)≈0)
# 			return [Temp]
# 		end
# 	end
# 	return coco_zeros(φ, l, m, μ, T, A)
# end


# --------------- SphericalMatrix

struct SphericalMatrix{L1, L2, LEN1, LEN2, T} <: AbstractProperty
   val::SMatrix{LEN1, LEN2, T}
   _valL1::Val{L1}
   _valL2::Val{L2}
end

# differentiation - cf #27
# actually this here appears to be the generic form how to do the
# differentiation for arbtirary order tensors.
*(φ::SphericalMatrix{L1, L2, LEN1, LEN2}, dAA::SVector{N}
      ) where {L1, L2, LEN1, LEN2, N} =
      reshape(φ.val[:] * dAA', Size(LEN1, LEN2, N))

getL(φ::SphericalMatrix{L1,L2}) where {L1,L2} = L1, L2

# L = 0 -> (0,0)
# L = 1 -> (0,0), (1,-1), (1,0), (1,1)  -> 4
# L = 3 ->  ... + 5 -> 9
# 1 + 3 + 5 + ... + 2*L+1
# = L + 2 * (1 + ... + L) = L+1 + 2 * L * (L+1) / 2 = (L+1)^2
function SphericalMatrix(L1::Integer, L2::Integer; T = Float64)
   LEN1 = 2L1+1   # length of SH basis up to L
   LEN2 = 2L2+1
   return SphericalMatrix( zero(SMatrix{LEN1, LEN2, T}), Val{L1}(), Val{L2}() )
end

function SphericalMatrix{L1, L2, LEN1, LEN2, T}(x::AbstractMatrix) where {L1, L2, LEN1, LEN2, T}
   @assert size(x) == (LEN1, LEN2)
   SphericalMatrix{L1, L2, LEN1, LEN2, T}( SMatrix{LEN1, LEN2, T}(x...), Val(L1), Val(L2) )
end

SphericalMatrix{L1, L2, LEN1, LEN2, T}()  where {L1, L2, LEN1, LEN2, T} =
      SphericalMatrix( zero(SMatrix{LEN1, LEN2, T}), Val{L1}(), Val{L2}() )

filter(φ::SphericalMatrix, b::Array) = ( length(b) < 1 ? true :
        ( ( iseven(sum(bi.l for bi in b)) == iseven(sum(getL(φ))) ) &&
         ( abs(sum(bi.m for bi in b)) <= sum(getL(φ)) )  ) )

rot3Dcoeffs(::SphericalMatrix, T::DataType=Float64) = Rot3DCoeffs(T)

function mat_cou_coe(rotc::Rot3DCoeffs{T},
					   ll::StaticVector{N},
	               mm::StaticVector{N},
					   μμ::StaticVector{N},
					   L1::Integer, L2::Integer,
					   a::Integer, b::Integer) where {T,N}
	#@show ll,mm,μμ
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
			MM = [μμ; ma[i]; mb[j]]
			KK = [mm; μa[i]; μb[j]]
			Z[i,j] = Dpa[i].sign * Dqb[j].sign * rotc(LL, MM, KK).val
			#@show Z
		end
	end
	return SphericalMatrix(SMatrix{2L1+1,2L2+1,ComplexF64}(Z), Val{L1}(), Val{L2}())
end

 mat_cou_coe(rotc::Rot3DCoeffs{T},
 				ll, mm, μμ, L1::Integer, L2::Integer,
 				a::Integer, b::Integer) where {T,N} =
 				mat_cou_coe(rotc, SVector(ll...), SVector(mm...), SVector(μμ...), L1::Integer, L2::Integer,
 								a::Integer, b::Integer)

function _select_ab(φ::SphericalMatrix{L1,L2}, ll, mm, kk) where {L1,L2}
	M = sum(mm)
   K = sum(kk)
   Dp = rotation_D_matrix_ast(L1)
   Dq = rotation_D_matrix(L2)
	num_ab = 0
   list_ab = []
   for a = 1:2L1+1
      for b = 1:2L2+1
         Dpa = Dp[:,a]
         Dqb = Dq[b,:]
         ma = [Dpa[k].m for k in 1:2L1+1]
         mb = [Dqb[k].m for k in 1:2L2+1]
         μa = [Dpa[k].μ for k in 1:2L1+1]
         μb = [Dqb[k].μ for k in 1:2L2+1]
         msum = vec([ma[i]+mb[j] for i = 1:2L1+1, j = 1:2L2+1])
         μsum = vec([μa[i]+μb[j] for i = 1:2L1+1, j = 1:2L2+1])
         if prod(μsum.+M)==0 && prod(msum.+K)==0
            list_ab = [list_ab;(a,b)]
            num_ab = num_ab+1
         end
      end
   end
   if list_ab == []
      return false, 0
   else
      return list_ab, num_ab
   end
end

function coco_init(φ::SphericalMatrix{L1,L2}, l, m, μ, T, A) where{L1,L2}
   list, num = _select_ab(φ,SVector(l),SVector(m),SVector(μ))
   if num == 0
      @warn("IS IT POSSIBLE???")
      return SphericalMatrix{L1,L2,2L1+1,2L2+1,ComplexF64}()
   else
      if iseven(l[1] + L1 + L2) && abs(m[1]) <= L1+L2 && abs(μ[1]) <= L1+L2
         Temp = []
         for (a,b) in list
		      Temp = [Temp; mat_cou_coe(__rotcoeff_inv, SVector(l), SVector(m), SVector(μ), L1, L2, a, b)]
         end
         return vec([i for i in Temp])
	   else
		   Temp = [SphericalMatrix{L1,L2,2L1+1,2L2+1,ComplexF64}() for a=1:num]
         return vec([i for i in Temp])
		end
   end
end

coco_zeros(φ::TP, ll, mm, kk, T, A) where{TP <: SphericalMatrix} =
            zeros(TP, _select_ab(φ,ll,mm,kk)[2])

coco_filter(φ::SphericalMatrix{L1,L2}, ll, mm) where {L1,L2} =
            iseven(sum(ll)) == iseven(L1+L2) && (abs(sum(mm)) <=  L1+L2)

coco_filter(φ::SphericalMatrix{L1,L2}, ll, mm, kk) where {L1,L2} =
            iseven(sum(ll)) == iseven(L1+L2) &&
            (abs(sum(mm)) <=  L1+L2) &&
            (abs(sum(kk)) <=  L1+L2)

coco_dot(u1::SphericalMatrix, u2::SphericalMatrix) =
		dot(u1.val, u2.val)
