
import Base: -, +, *, filter, real, complex
import LinearAlgebra: norm, promote_leaf_eltypes

_basetype(φ::AbstractProperty) = Base.typename(typeof(φ)).wrapper

# TODO: rewrite all this with meta-programming 

@inline +(φ1::AbstractProperty, φ2::AbstractProperty) = (_basetype(φ1))( φ1.val + φ2.val )
@inline -(φ1::AbstractProperty, φ2::AbstractProperty) = (_basetype(φ1))( φ1.val - φ2.val )
@inline -(φ1::AbstractProperty) = (_basetype(φ1))( -φ1.val )
@inline *(φ1::AbstractProperty, φ2::AbstractProperty) = (_basetype(φ1))( φ1.val * φ2.val )

@inline *(a::Union{Number, AbstractMatrix}, φ::AbstractProperty) = (_basetype(φ))(a * φ.val)
@inline *(φ::AbstractProperty, a::Union{Number, AbstractMatrix})  = (_basetype(φ))(φ.val * a)

*(a::AbstractVector{<: Number}, φ::AbstractProperty) = a .* Ref(φ)

Base.isapprox(φ1::AbstractProperty, φ2::AbstractProperty) = isapprox(φ1.val, φ2.val)

@inline norm(φ::AbstractProperty) = norm(φ.val)
@inline Base.length(φ::AbstractProperty) = length(φ.val)
@inline Base.size(φ::AbstractProperty) = size(φ.val)
@inline Base.zero(φ::AbstractProperty) = (_basetype(φ))(zero(φ.val))
@inline Base.zero(::Type{T}) where {T <: AbstractProperty} = zero(T())

promote_leaf_eltypes(φ::T) where {T <: AbstractProperty} = promote_leaf_eltypes(φ.val)

Base.convert(T::Type{TP}, φ::TP) where {TP <: AbstractProperty} = φ
Base.convert(T::Type, φ::AbstractProperty) = convert(T, φ.val)
Base.convert(T::Type{Any}, φ::AbstractProperty) = φ

# TODO: this was a hack; not clear this is a good idea ... 
#       remove during next cleanup and see what happens? 
Base.iterate(φ::AbstractProperty) = φ, nothing
Base.iterate(φ::AbstractProperty, ::Nothing) = nothing



"""
`coco_o_daa` : implements a tensor product between a coupling coefficient 
(usually an `AbstractProperty`) and a gradient (usually a `DState`). 
"""
function coco_o_daa(φ::AbstractProperty, b::TX) where {TX <: XState}
   SYMS = _syms(TX)
   vals = ntuple( i -> coco_o_daa(φ.val, _x(b)[SYMS[i]]), length(SYMS) )
   return TX( NamedTuple{SYMS}(vals) )
end

coco_o_daa(cc::SVector{N, <: AbstractProperty}, b::TX) where {N, TX <: XState} = 
      SVector( ntuple(i -> coco_o_daa(cc[i], b), N) )

coco_o_daa(cc::Number, b::Number) = cc * b
coco_o_daa(cc::Number, b::SVector) = cc * b
coco_o_daa(cc::SVector, b::SVector) = cc * transpose(b)
coco_o_daa(cc::SMatrix{N1,N2}, b::SVector{N3}) where {N1,N2,N3} =
		reshape(cc[:] * transpose(b), Size(N1, N2, N3))
coco_o_daa(cc::SArray{Tuple{N1,N2,N3}}, b::SVector{N4}) where {N1,N2,N3,N4} =
		reshape(cc[:] * transpose(b), Size(N1, N2, N3, N4))


# TODO: is this needed or can it be removed? 
#       maybe it should also be allowed for a DState?
#       it is also more of an ⊗ ... revisit during next cleanup 
*(φ::AbstractProperty, b::AbstractState) = coco_o_daa(φ, b)


# default behaviour - overwritten e.g. for EuclideanVector where 
# cocos are always complex independently of whether phi is real.
coco_type(T::Type{<: AbstractProperty}) = T


"""
`struct Invariant{D}` : specifies that the output of an ACE is
an invariant scalar.
"""
struct Invariant{T} <: AbstractProperty
   val::T
end

Base.show(io::IO, φ::Invariant) = print(io, "i($(φ.val))")

Base.one(φ::Invariant{T}) where {T} = Invariant(one(T))
Base.one(::Type{Invariant{T}}) where {T} = Invariant(one(T))

isrealB(::Invariant{<: Real}) = true 
isrealB(::Invariant{<: Complex}) = false 
isrealAA(::Invariant{<: Real}) = true 
isrealAA(::Invariant{<: Complex}) = false 

Invariant{T}() where {T <: Number} = Invariant{T}(zero(T))

Invariant(T::DataType = Float64) = Invariant{T}()

real(φ::Invariant) = Invariant(real(φ.val))
complex(φ::Invariant) = Invariant(complex(φ.val))
complex(::Type{Invariant{T}}) where {T} = Invariant{complex(T)}
complex(φ::AbstractVector{<: Invariant}) = complex.(φ)
+(φ::Invariant, x::Number) = Invariant(φ.val + x)
+(x::Number, φ::Invariant) = Invariant(φ.val + x)

Base.convert(::Type{Invariant{T}}, x::Number) where {T} = Invariant(convert(T, x))

*(φ1::Invariant, φ2::Invariant) = Invariant(φ1.val * φ2.val)

write_dict(φ::Invariant{T})  where {T} =
   Dict("__id__" => "ACE_Invariant",
        "val" => φ.val,
        "T" => write_dict(T) )

read_dict(::Val{:ACE_Invariant}, D::Dict) =
      Invariant{read_dict(D["T"])}(D["val"])


function filter(φ::Invariant, grp::O3, b::Array) 
   if length(b) <= 1
      return true 
   end 
   suml = sum( getl(grp, bi) for bi in b )
   if haskey(b[1], msym(grp))  # depends on context whether m come along?
      summ = sum( getm(grp, bi) for bi in b )
      return iseven(suml) && iszero(summ)
   end
   return iseven(suml)   
end

filter(φ::Invariant, grp::O3O3, b::Array) = 
      filter(φ, grp.G1, b) && filter(φ, grp.G2, b)
      
filter(φ::AbstractProperty, grp::NoSym, b::Array) = true

rot3Dcoeffs(::Invariant, T=Float64) = Rot3DCoeffs(T)

# TODO: this is a naive implementation of differentiation.
#       cf https://github.com/ACEsuit/ACE.jl/issues/27
#       for further discussion
*(φ::Invariant, dAA::SVector) = φ.val * dAA

coco_init(::Invariant{T}) where {T} = [ Invariant(complex(T(1)));; ] 

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

Base.show(io::IO, φ::EuclideanVector) = 
   print(io, "e[$(φ.val[1]), $(φ.val[2]), $(φ.val[3])]")

real(φ::EuclideanVector) = EuclideanVector(real.(φ.val))
complex(φ::EuclideanVector) = EuclideanVector(complex(φ.val))
complex(::Type{EuclideanVector{T}}) where {T} = EuclideanVector{complex(T)}

+(x::SVector{3}, y::EuclideanVector) = EuclideanVector(x + y.val)
Base.convert(::Type{SVector{3, T}}, φ::EuclideanVector) where {T} = convert(SVector{3, T}, φ.val)

isrealB(::EuclideanVector{T}) where {T} = (T == real(T))
isrealAA(::EuclideanVector) = false

# CO: removed this, is it needed? 
# Base.getindex(φ::EuclideanVector, i::Integer) = φ.val[i] 

EuclideanVector{T}() where {T <: Number} = EuclideanVector{T}(zero(SVector{3, T}))
EuclideanVector(T::DataType=Float64) = EuclideanVector{T}()


function filter(φ::EuclideanVector, grp::O3, bb::Array)
   if length(bb) == 0  # no zero-correlations allowed 
      return false 
   end 
   if length(bb) == 1 #MS: Not sure if this should be here
                      # CO: good question - need to investigate 
      return true
   end
   suml = sum( getl(grp, bi) for bi in bb )
   if haskey(bb[1], msym(grp))  # depends on context whether m come along?
      summ = sum( getm(grp, bi) for bi in bb )
      return isodd(suml) && abs(summ) <= 1
   end
   return isodd(suml)
end

rot3Dcoeffs(::EuclideanVector,T=Float64) = Rot3DCoeffsEquiv{T,1}(Dict[], ClebschGordan(T))

write_dict(φ::EuclideanVector{T}) where {T} =
      Dict("__id__" => "ACE_EuclideanVector",
             "val" => write_dict(Vector(φ.val)) )

function read_dict(::Val{:ACE_EuclideanVector}, D::Dict)
   return EuclideanVector(SVector{3}(read_dict(D["val"])))
end

# differentiation - cf #27
# *(φ::EuclideanVector, dAA::SVector) = φ.val * dAA'
*(prop::EuclideanVector, c::SVector{N, T}) where {T<:Number,N} = SVector{N}(prop*c[i] for i=1:N)

coco_init(phi::EuclideanVector{CT}, l, m, μ, T, A) where {CT<:Real} = (
      (l == 1 && abs(m) <= 1 && abs(μ) <= 1)
         ? [EuclideanVector(conj(crmatrices[(l=l,m=-m,mu=-μ,i=i)])) for i=1:3]
         : coco_zeros(phi, l, m, μ, T, A)  )

#coco_init(phi::EuclideanVector{CT}, l, m, μ, T, A) where {CT<:Real} = (
#      (l == 1 && abs(m) <= 1 && abs(μ) <= 1)
#         ? [EuclideanVector(rmatrices[m,μ][:,k]) for k=1:3]
#         : coco_zeros(phi, l, m, μ, T, A)  )

# coco_init(phi::EuclideanVector{CT}
#   this is not needed, since the EuclideanVector should never give us 
#   a constant basis anyhow. Still ... this could become a problem if we 
#   ever want to artificially increase the AA basis in order to get some 
#   savings elsewhere. Maybe need to revisit this...

coco_type(φ::EuclideanVector) = typeof(complex(φ))
coco_type(::Type{EuclideanVector{T}}) where {T} = EuclideanVector{complex(T)}

coco_zeros(φ::EuclideanVector, ll, mm, kk, T, A) = zeros(typeof(complex(φ)), 3)

coco_filter(::EuclideanVector, ll, mm) =
            isodd(sum(ll)) && (abs(sum(mm)) <= 1)

coco_filter(::EuclideanVector, ll, mm, kk) =
      abs(sum(mm)) <= 1 &&
      abs(sum(kk)) <= 1 &&
      isodd(sum(ll))

coco_dot(u1::EuclideanVector, u2::EuclideanVector) = dot(u1.val, u2.val)

include("eucl/cov_coeffs_dict.jl")

#---------------------- Equivariant matrices

abstract type AbstractEuclideanMatrix{T} <: AbstractProperty end

real(φ::E) where { E <: AbstractEuclideanMatrix} = E(real.(φ.val))

+(x::SMatrix{3}, y::E) where { E <: AbstractEuclideanMatrix} = E(x + y.val)

Base.convert(::Type{SMatrix{3, 3, T, 9}}, φ::AbstractEuclideanMatrix) where {T}  =  convert(SMatrix{3, 3, T, 9}, φ.val)

isrealAA(::AbstractEuclideanMatrix)  = false

function filter(::AbstractEuclideanMatrix, grp::O3, bb::Array)
   if length(bb) == 0  # no zero-correlations allowed 
      return false 
   end
   if length(bb) == 1 #MS: Not sure if this should be here
      return true
   end
   suml = sum( getl(grp, bi) for bi in bb )
   if haskey(bb[1], msym(grp))  # depends on context whether m come along?
      summ = sum( getm(grp, bi) for bi in bb )
      return iseven(suml) && abs(summ) <= 2
   end
   return iseven(suml)
end

rot3Dcoeffs(::AbstractEuclideanMatrix,T=Float64) = Rot3DCoeffsEquiv{T,1}(Dict[], ClebschGordan(T))

coco_type(φ::AbstractEuclideanMatrix) = typeof(complex(φ))

coco_zeros(φ::AbstractEuclideanMatrix, ll, mm, kk, T, A) =  zeros(typeof(complex(φ)), 9)

coco_filter(::AbstractEuclideanMatrix, ll, mm) = iseven(sum(ll)) && (abs(sum(mm)) <= 2)

coco_filter(::AbstractEuclideanMatrix, ll, mm, kk) =  abs(sum(mm)) <= 2 && 
                                 abs(sum(kk)) <= 2 &&
                                 iseven(sum(ll))
coco_dot(u1::AbstractEuclideanMatrix, u2::AbstractEuclideanMatrix) = sum(transpose(conj.( u1.val)) * u2.val)

*(prop::ACE.AbstractEuclideanMatrix, c::SVector{N, T}) where {T<:Number,N} = SVector{N}(prop*c[i] for i=1:N)

function Base.show(io::IO, φ::AbstractEuclideanMatrix)
   # println(io, "3x3 $(typeof(φ)):")
   println(io, "$(_type_marker(φ))[ $(φ.val[1,1]), $(φ.val[1,2]), $(φ.val[1,3]);")
   println(io, "   $(φ.val[2,1]), $(φ.val[2,2]), $(φ.val[2,3]);")
   print(io,   "   $(φ.val[3,1]), $(φ.val[3,2]), $(φ.val[3,3]) ]")
end

include("eucl/equi_coeffs_dict.jl")

# Equivariant 3 x 3 Euclidean Matrix of general form (i.e., doesn't need to be symmetric or anti-symmetric)
struct EuclideanMatrix{T} <: AbstractEuclideanMatrix{T}
   val::SMatrix{3, 3, T, 9}
end

_type_marker(φ::EuclideanMatrix) = "e"

coco_init(phi::EuclideanMatrix{CT}, l, m, μ, T, A) where {CT<:Real} = (
   (l <= 2 && abs(m) <= l && abs(μ) <= l)
      ? vec([EuclideanMatrix(conj.(mrmatrices[(l=l,m=-m,mu=-μ,i=i,j=j)])) for i=1:3 for j=1:3])
      : coco_zeros(phi, l, m, μ, T, A)  )


write_dict(φ::EuclideanMatrix{T}) where {T} =
      Dict("__id__" => "ACE_EuclideanMatrix",
              "valr" => write_dict(real.(Matrix(φ.val))),
              "vali" => write_dict(imag.(Matrix(φ.val))),
                "T" => write_dict(T) )

function read_dict(::Val{:ACE_EuclideanMatrix}, D::Dict)
   T = read_dict(D["T"])
   valr = SMatrix{3, 3, T, 9}(read_dict(D["valr"]))
   vali = SMatrix{3, 3, T, 9}(read_dict(D["vali"]))
   return EuclideanMatrix{T}(valr + im * vali)
end

# ---------------------
# Equivariant 3 x 3 symmetric Euclidean Matrix 


struct SymmetricEuclideanMatrix{T} <:  AbstractEuclideanMatrix{T} #where {S<:MatrixSymmetry}
   val::SMatrix{3, 3, T, 9}
end

_type_marker(φ::SymmetricEuclideanMatrix) = "se"

function coco_init(phi::SymmetricEuclideanMatrix{CT}, l, m, μ, T, A) where {CT<:Real}
   return ( ( (l == 2 && abs(m) <= 2 && abs(μ) <= 2) || (l == 0 && abs(m) == 0 && abs(μ) == 0) )
      ? vec([SymmetricEuclideanMatrix(conj.(mrmatrices[(l=l,m=-m,mu=-μ,i=i,j=j)])) for i=1:3 for j=1:3])
      : coco_zeros(phi, l, m, μ, T, A)  )
end

function ACE.write_dict(φ::SymmetricEuclideanMatrix{T}) where {T}
   Dict("__id__" => "ACE_SymmetricEuclideanMatrix",
         "valr" => write_dict(real.(Matrix(φ.val))),
         "vali" => write_dict(imag.(Matrix(φ.val))),
            "T" => write_dict(T))         
end 

function ACE.read_dict(::Val{:ACE_SymmetricEuclideanMatrix}, D::Dict)
   T = read_dict(D["T"])
   valr = SMatrix{3, 3, T, 9}(read_dict(D["valr"]))
   vali = SMatrix{3, 3, T, 9}(read_dict(D["vali"]))
   return SymmetricEuclideanMatrix{T}(valr + im * vali)
end

# Equivariant 3 x 3 anti-symmetric Euclidean Matrix 
# struct AntiSymmetricEuclideanMatrix{T} <: AbstractEuclideanMatrix{T} #where {S<:MatrixSymmetry}
#    val::SMatrix{3, 3, T, 9}
# end

# _type_marker(φ::AntiSymmetricEuclideanMatrix) = "ase"

# function coco_init(phi::AntiSymmetricEuclideanMatrix{CT}, l, m, μ, T, A) where {CT<:Real}
#    return ( (l == 1 && abs(m) <= 1 && abs(μ) <= 1)
#       ? vec([AntiSymmetricEuclideanMatrix(conj.(mrmatrices[(l=l,m=-m,mu=-μ,i=i,j=j)])) for i=1:3 for j=1:3])
#       : coco_zeros(phi, l, m, μ, T, A)  )
# end

# function ACE.write_dict(φ::AntiSymmetricEuclideanMatrix{T}) where {T}
#    Dict("__id__" => "ACE_AntiSymmetricEuclideanMatrix",
#          "valr" => write_dict(real.(Matrix(φ.val))),
#          "vali" => write_dict(imag.(Matrix(φ.val))),
#             "T" => write_dict(T))         
# end 

# function ACE.read_dict(::Val{:ACE_AntiSymmetricEuclideanMatrix}, D::Dict)
#    T = read_dict(D["T"])
#    valr = SMatrix{3, 3, T, 9}(read_dict(D["valr"]))
#    vali = SMatrix{3, 3, T, 9}(read_dict(D["vali"]))
#    return AntiSymmetricEuclideanMatrix{T}(valr + im * vali)
# end

#  for E = (:EuclideanMatrix,:SymmetricEuclideanMatrix, :AntiSymmetricEuclideanMatrix)
for E = (:EuclideanMatrix,:SymmetricEuclideanMatrix)
    eval(
       quote
 
       complex(φ::$E) = $E(complex(φ.val))
       complex(::Type{$E{T}}) where {T} = $E{complex(T)}
    
       Base.convert(::Type{$E{T}}, φ::$E{T}) where {T<:Number} = $E(φ.val) #This function was previously not defined for EuclideanMatrix 
       isrealB(::$E{T}) where {T} = (T == real(T))
 
       $E{T}() where {T <: Number} = $E{T}(zero(SMatrix{3, 3, T, 9}))
       $E(T::DataType=Float64) = $E{T}()
       coco_type(::Type{$E{T}}) where {T} = $E{complex(T)}
         
       Base.promote_rule(::Type{T1}, ::Type{$E{T2}}
             ) where {T1 <: Number, T2 <: Number} = 
       $E{promote_type(T1, T2)}

       end
    ) 
 end
  

 Base.promote_rule(t1::Type{T1}, t2::Type{SMatrix{N,N, T2}}
                        ) where {N, T1 <: Number, T2 <: AbstractProperty} = 
            SMatrix{N, N, promote_rule(t1, T2)}

# --------------------- 
# some codes to help convert from svector, smatrix to spherical ... 

@generated function __getL(::Val{N}) where {N} 
   @assert isodd(N)
   quote
      $( div(N-1, 2) )
   end
end

__getL(::SVector{N}) where {N} = __getL(Val{N}())

__getL1L2(::SMatrix{N1, N2}) where {N1, N2} = __getL(Val{N1}()), __getL(Val{N2}())

# --------------------- SphericalVector

# TODO: remove the Val{L} - not needed (next cleanup)
struct SphericalVector{L, LEN, T} <: AbstractProperty
   val::SVector{LEN, T}
   _valL::Val{L}
end

SphericalVector(val::SVector) = SphericalVector(val, Val(__getL(val)))


# # differentiation - cf #27
# *(φ::SphericalVector, dAA::SVector) = φ.val * dAA'

isrealB(::SphericalVector) = false 
isrealAA(::SphericalVector) = false 


real(φ::SphericalVector) = SphericalVector(real(φ.val), φ._valL)

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
   SphericalVector{L, LEN, T}( SVector{LEN, T}(x), Val(L) )
end

SphericalVector{L, LEN, T}()  where {L, LEN, T} =
      SphericalVector( zero(SVector{LEN, T}), Val{L}() )

function filter(φ::SphericalVector, grp::O3, b::Array)
	if length(b) <= 1
		return true
	end
	suml = sum( getl(grp, bi) for bi in b )
   if haskey(b[1], msym(grp))
      summ = sum( getm(grp, bi) for bi in b )
      return iseven(suml) == iseven(getL(φ)) && abs(summ) <= getL(φ)
   end
   return iseven(suml) == iseven(getL(φ))
end

rot3Dcoeffs(::SphericalVector, T::DataType=Float64) = Rot3DCoeffs(T)


const __rotcoeff_inv = Rotations3D.Rot3DCoeffs(Invariant())

using ACE.Wigner: wigner_D_indices

# Equation (1.2) - vector value coupling coefficients
# ∫_{SO3} D^{ll}_{μμmm} D^*(Q) e^t dQ -> 2L+1 column vector
function vec_cou_coe(rotc::Rot3DCoeffs{T},
					      l::Integer, m::Integer, μ::Integer,
					      L::Integer, t::Integer) where {T,N}
	@assert 0 < t <= 2L+1
	D = wigner_D_indices(L)'   # Dt = D[:,t]  -->  # D^* ⋅ e^t
	LL = SA[l, L]
	Z = ntuple(i -> begin
			cc = (rotc(LL, SA[μ, D[i, t].m], SA[m, D[i, t].μ]).val)::T
			D[i, t].sign * cc
		end, 2*L+1)
	return SphericalVector{L, 2L+1, Complex{T}}(SVector(Z))
end

function _select_t(φ::SphericalVector{L}, l, M, K) where {L}
	D = wigner_D_indices(L)'
	tret = -1; numt = 0
	for t = 1:2L+1
		prodμt = prod( (D[i, t].μ + M) for i in 1:2L+1)  # avoid more allocations
		prodmt = prod( (D[i, t].m + K) for i in 1:2L+1)
		if prodμt == prodmt == 0
			tret = t; numt += 1
		end
	end
	# We assumed that there is only one coefficient; this will warn us if it fails
	@assert numt == 1
	return tret
end


coco_zeros(φ::TP, ll, mm, kk, T, A)  where {TP <: SphericalVector} = zero(TP)

coco_dot(u1::SphericalVector, u2::SphericalVector) =
		dot(u1.val, u2.val)

coco_filter(φ::SphericalVector{L}, ll, mm) where {L} =
		iseven(sum(ll) + L) && (abs(sum(mm)) <=  L)

coco_filter(φ::SphericalVector{L}, ll, mm, kk) where {L} =
      iseven(sum(ll) + L) && (abs(sum(mm)) <=  L) && (abs(sum(kk)) <= L)


coco_init(φ::SphericalVector{L}, l, m, μ, T, A) where {L} =
			vec_cou_coe(__rotcoeff_inv, l, m, μ, L, _select_t(φ, l, m, μ))

coco_init(φ::SphericalVector{L}) where {L} = 
            L==0 ? reshape([SphericalVector(SVector(1.0+0.0im))],1,1) : []

# --------------- SphericalMatrix

struct SphericalMatrix{L1, L2, LEN1, LEN2, T, LL} <: AbstractProperty
   val::SMatrix{LEN1, LEN2, T, LL}
   _valL1::Val{L1}
   _valL2::Val{L2}
end

function Base.show(io::IO, φ::Union{SphericalVector, SphericalMatrix})
   buffer = IOBuffer()
   print(buffer, round.(φ.val, digits=4))
   str = String(take!(buffer))
   lenn = findfirst('[', str)
   print(io, "y" * str[lenn:end])
end

SphericalMatrix(val::SMatrix) = SphericalMatrix(val, Val.(__getL1L2(val))...)

# differentiation - cf #27
# actually this here appears to be the generic form how to do the
# differentiation for arbtirary order tensors.
# *(φ::SphericalMatrix{L1, L2, LEN1, LEN2}, dAA::SVector{N}
#       ) where {L1, L2, LEN1, LEN2, N} =
#       reshape(φ.val[:] * dAA', Size(LEN1, LEN2, N))

getL(φ::SphericalMatrix{L1,L2}) where {L1,L2} = L1, L2

isrealB(::SphericalMatrix) = false 
isrealAA(::SphericalMatrix) = false 


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

# this is a mess - need to fix it, we should need just one constructor???

function SphericalMatrix{L1, L2, LEN1, LEN2, T, LL}(x::AbstractMatrix) where {L1, L2, LEN1, LEN2, T, LL}
   @assert size(x) == (LEN1, LEN2)
   SphericalMatrix{L1, L2, LEN1, LEN2, T, LL}( SMatrix{LEN1, LEN2, T}(x), Val(L1), Val(L2) )
end

SphericalMatrix{L1, L2, LEN1, LEN2, T, LL}()  where {L1, L2, LEN1, LEN2, T, LL} =
      SphericalMatrix( zero(SMatrix{LEN1, LEN2, T}), Val{L1}(), Val{L2}() )

SphericalMatrix{L1, L2, LEN1, LEN2, T}()  where {L1, L2, LEN1, LEN2, T, LL} =
		SphericalMatrix( zero(SMatrix{LEN1, LEN2, T}), Val{L1}(), Val{L2}() )

function filter(φ::SphericalMatrix, grp::O3, b::Array)
	if length(b) < 1
		return true
	end
	suml = sum( getl(grp, bi) for bi in b )
   if haskey(b[1], msym(grp))
      summ = sum( getm(grp, bi) for bi in b )
      return iseven(suml) == iseven( sum(getL(φ)) ) && abs(summ) <= sum(getL(φ))
   end
   return iseven(suml) == iseven( sum(getL(φ)) )
end

rot3Dcoeffs(::SphericalMatrix, T::DataType=Float64) = Rot3DCoeffs(T)

function mat_cou_coe(rotc::Rot3DCoeffs{T},
				   		l::Integer, m::Integer, μ::Integer,
					   	a::Integer, b::Integer,
							::Val{L1}, ::Val{L2}) where {T, L1, L2}
	@assert (0 < a <= 2L1 + 1) && (0 < b <= 2L2 + 1)
	Z = zero(MMatrix{2L1+1, 2L2+1, Complex{T}})  # zeros(2 * L1 + 1, 2 * L2 + 1)
	Dp = wigner_D_indices(L1)'
	Dq = wigner_D_indices(L2)
	LL = SA[l, L1, L2]
	for i = 1:(2 * L1 + 1)
		for j = 1:(2 * L2 + 1)
			MM = SA[μ, Dp[i,a].m, Dq[b,j].m]
			KK = SA[m, Dp[i,a].μ, Dq[b,j].μ]
			cc = (rotc(LL, MM, KK).val)::T
			Z[i,j] = Dp[i,a].sign * Dq[b,j].sign * cc
		end
	end
	return SphericalMatrix(SMatrix(Z), Val{L1}(), Val{L2}())
	# return SphericalMatrix(SMatrix{2L1+1,2L2+1,Complex{T}}(Z), Val{L1}(), Val{L2}())
end


function _select_ab(φ::SphericalMatrix{L1,L2}, M, K) where {L1,L2}
   Dp = wigner_D_indices(L1)'
   Dq = wigner_D_indices(L2)
   list_ab = Tuple{Int, Int}[]
   for a = 1:2L1+1
      for b = 1:2L2+1
			# pm = prod( ma[i] + mb[j] + K for i = 1:2L1+1, j = 1:2L2+1)
			pm = prod( Dp[i,a].m + Dq[b,j].m + K for i = 1:2L1+1, j = 1:2L2+1)
			# pμ = prod(μa[i] + μb[j] + M for i = 1:2L1+1, j = 1:2L2+1)
			pμ = prod( Dp[i,a].μ + Dq[b,j].μ + M for i = 1:2L1+1, j = 1:2L2+1)
         if pμ == pm ==0
				push!(list_ab, (a,b))
         end
      end
   end
	return list_ab
end


function coco_init(φ::SphericalMatrix{L1,L2}, l, m, μ, T, A) where{L1,L2}
   list = _select_ab(φ, m, μ)
	@assert length(list) > 0
	# MAIN CASE
   if abs(m) <= L1+L2 && abs(μ) <= L1+L2
		return [ mat_cou_coe(__rotcoeff_inv, l, m, μ, a, b, Val(L1), Val(L2))
				   for (a,b) in list ]
	end

	# @warn("SHOULDN'T BE HERE!!")
   return fill( zero(typeof(φ)), length(list) )
end

coco_init(φ::SphericalMatrix{L1,L2}) where{L1,L2} = 
				  L1==L2 ? reshape([ACE.SphericalMatrix(SMatrix{2L1+1,2L2+1,ComplexF64}(I(2L1+1)),Val(L1),Val(L2))],1,1) : []

coco_zeros(φ::TP, ll, mm, kk, T, A) where{TP <: SphericalMatrix} =
            zeros(TP, length(_select_ab(φ, sum(mm), sum(kk))))

coco_filter(φ::SphericalMatrix{L1,L2}, ll, mm) where {L1,L2} =
            iseven(sum(ll)) == iseven(L1+L2) && (abs(sum(mm)) <=  L1+L2)

coco_filter(φ::SphericalMatrix{L1,L2}, ll, mm, kk) where {L1,L2} =
            iseven(sum(ll) + L1 + L2) &&
            (abs(sum(mm)) <=  L1+L2) &&
            (abs(sum(kk)) <=  L1+L2)

coco_dot(u1::SphericalMatrix, u2::SphericalMatrix) =
		dot(u1.val, u2.val)


#  Some promotion rules 

Base.promote_rule(::Type{T1}, ::Type{Invariant{T2}}
                  ) where {T1 <: Number, T2 <: Number} = 
      Invariant{promote_type(T1, T2)}

Base.promote_rule(::Type{T1}, ::Type{EuclideanVector{T2}}
                  ) where {T1 <: Number, T2 <: Number} = 
      EuclideanVector{promote_type(T1, T2)}

Base.promote_rule(t1::Type{T1}, t2::Type{SVector{N, T2}}
                  ) where {N, T1 <: Number, T2 <: AbstractProperty} = 
      SVector{N, promote_rule(t1, T2)}

# --------------------------- AD related codes 

# an x -> x.val implementation with custom adjoints to sort out the 
# mess created by the AbstractProperties
# maybe this feels a bit wrong, definitely a hack. What might be nicer 
# is to introduce a "Dual Property" similar to the "DState"; Then we 
# could have something along the lines of  DProp * Prop = scalar or 
# _contract(DProp, Prop) = scalar; That would be the "systematic" and 
# "disciplined" way of implementing this. 

"""
`val(x) = x.val`, normally to be used if x is a property. This should be used 
instead of x.val when the operation is part of a bigger expression that is 
to be ADed. I.e. `val` has rrules implemented that should allow taking up 
to two derivatives. 

TODO: at the moment this is a bit hacky, and needs to be adjusted over time
as we learn more about how to best implement AD. The real question is what 
is the correct adjoint of this operation? E.g., if 
````
     val : { Invariant{T} } -> { T }
```
then should 
```
   val* :  { T } -> { Invariant{T} }
```
? If this is the case, then we need a parameterised `val` function. I.e. 
we need to remember what the Property is that we started from.
"""
val(x) = x.val 

function _rrule_val(dp, x)     # ∂/∂x (dp * x) = dp 
   @assert dp isa Number 
   return NoTangent(), dp
end

rrule(::typeof(val), x) = 
         val(x), 
         dp -> _rrule_val(dp, x)

function rrule(::typeof(_rrule_val), dp, x)   # ∂/∂... (0 + dp * dq[2])
      @assert dp isa Number 
      function second_adj(dq)
         @assert dq[1] == ZeroTangent() 
         # @assert dq[2] isa Number  # TODO -> revisit this?!
         return NoTangent(), dq[2], ZeroTangent()
      end
      return _rrule_val(dp, x), second_adj
end 

import ChainRulesCore: ProjectTo

(::ProjectTo{T})(φ::Invariant{T}) where {T} = val(φ)



