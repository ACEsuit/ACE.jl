using ACE, StaticArrays
using Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, SymmetricBasis, PIBasis, EuclideanMatrix
using ACE.Random: rand_rot, rand_refl, rand_vec3
using ACEbase.Testing: fdtest
using ACEbase.Testing: println_slim
# construct the 1p-basis
maxdeg = 5
ord = 2
Bsel = SimpleSparseBasis(ord, maxdeg)

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)

# generate a configuration
nX = 10
_randX() = State(rr = rand_vec3(B1p["Rn"]))
Xs = [ _randX() for _=1:nX ] 
cfg = ACEConfig(Xs)



# ACE.EuclideanMatrix{T}() where {T <: Number} = ACE.EuclideanMatrix{T,Val{:general}}(zero(SMatrix{3, 3, T, 9}), :general, Val(:general))
# ACE.EuclideanMatrix(T::DataType=Float64) = ACE.EuclideanMatrix{T}()
# ACE.EuclideanMatrix(T::DataType, symmetry::Symbol) = ACE.EuclideanMatrix(zero(SMatrix{3, 3, T, 9}), symmetry, Val(symmetry))
# ACE.EuclideanMatrix(val::SMatrix{3, 3, T, 9}) where {T <: Number} = ACE.EuclideanMatrix(val, :general,Val(:general)) # should depend on symmetry of val
# ACE.EuclideanMatrix(val::SMatrix{3, 3, T, 9}, symmetry::Symbol) where {T <: Number} = ACE.EuclideanMatrix{T,Val{symmetry}}(val, symmetry,Val(symmetry))

# Needs to take care of symbol
# ACE.EuclideanMatrix{T,S}() where {T <: Number,S} = ACE.EuclideanMatrix{T,S}(zero(SMatrix{3, 3, T, 9}), :general, S())

# Base.convert(::Type{EuclideanMatrix{T, Val{:symmetric}}}, φ::EuclideanMatrix{T, Val{:general}}) where {T<:Number} = ACE.EuclideanMatrix(φ.val,:symmetric)

# function Base.convert(T::Type, φ::ACE.AbstractProperty)
#    @show T
#    @show φ
#    @show typeof(φ)
#    convert(T, φ.val)
# end
# ACE.coco_type(::Type{EuclideanMatrix{T,S}}) where {T,S} = EuclideanMatrix{complex(T),S}

# ACE.EuclideanMatrix{Float64,Val{:general}}(zero(SMatrix{3, 3, Float64, 9}), :general, Val(:general))

# convert(ACE.EuclideanMatrix{Float64, Val{:general}}, zero(SMatrix{3, 3, ComplexF64, 9}))

# ACE.EuclideanMatrix{Float64,Val{:general}}() 
# ACE.EuclideanMatrix{Float64}()
# a= ACE.EuclideanMatrix(zero(SMatrix{3, 3, Float64, 9}), :general, Val(:general))
# ACE.coco_type(a) 
# ACE.coco_type(EuclideanMatrix{Float64})
# ACE.coco_type(EuclideanMatrix{Float64,Val{:general}})
# ACE.coco_type(EuclideanMatrix{Float64,Val{:symmetric}})
# ACE.coco_type(EuclideanMatrix{Float64})


# ACE.coco_zeros(φ::EuclideanMatrix, ll, mm, kk, T, A) =  zeros(typeof(ACE.complex(φ)), 9)

# ACE.complex(φ::EuclideanMatrix{T,Val{symb}}) where {T,symb} = EuclideanMatrix(ACE.complex(φ.val), symb)

#ACE.complex(φ::EuclideanMatrix{T,Val{:general}}) where {T} = EuclideanMatrix(ACE.complex(φ.val),:general, Val(:general))
#ACE.complex(φ::EuclideanMatrix{T,Val{:antisymmetric}}) where {T} = EuclideanMatrix(ACE.complex(φ.val),:antisymmetric, Val(:antisymmetric))
#complex(::Type{EuclideanMatrix{T}}) where {T} = EuclideanMatrix{complex(T)}

# φ = ACE.EuclideanMatrix(Float64, :symmetric)
# typeof(φ)<: EuclideanMatrix{T,Val{symb}} where {T, symb}
# EuclideanMatrix(ACE.complex(φ.val),:symmetric)
# typeof(ACE.complex(φ))
# typeof(φ) 
# a = ACE.coco_zeros(φ, 1, 1, 1, 1, 1)
# typeof(a)
# a= ACE.EuclideanMatrix(zero(SMatrix{3, 3, Float64, 9}))
# typeof(a)
# a= ACE.EuclideanMatrix(Float64, :symmetric) 
# typeof(a)
# a = ACE.EuclideanMatrix{Float64}()
# typeof(a)
# a = ACE.EuclideanMatrix(zero(SMatrix{3, 3, Float64, 9}), :symmetric)
# typeof(a)

# v = ACE.Invariant()
# v + 1.0
# 1.0 + v
# v + ComplexF64(1.0)
# ACE.EuclideanVector(Float64) + (@SVector rand(3))
@info("SymmetricBasis construction and evaluation: EuclideanMatrix")
# φ = ACE.EuclideanMatrix(Float64,:general)
# typeof(φ)
# #Base.convert(T::Type, φ::AbstractProperty) = convert(T, φ.val)
# Base.convert(SMatrix{3, 3, Complex{Float64}, 9}, φ )
# convert(Type{SMatrix{3, 3, Complex{Float64}, 9}}, φ.val)

# φ =EuclideanMatrix(Float64,:symmetric)
# EuclideanMatrix(Float64,:symmetric) + (@SMatrix rand(3,3))
# typeof(φ)
 
# convert(Any, ACE.EuclideanVector(Float64))
# convert(typeof(ACE.EuclideanVector(Float64)), ACE.EuclideanVector(Float64))
# convert(SVector{3,Float64}, ACE.EuclideanVector(Float64))
# EuclideanMatrix{T}() where {T <: Number} = EuclideanMatrix{T}(zero(SMatrix{3, 3, T, 9}), Val(:general))
# φ =EuclideanMatrix{Float64}()
# EuclideanMatrix{Float64}(zero(SMatrix{3, 3, Float64, 9}), Val(:general))

for symmetry in [:general :symmetric]
   @info("Symmetry type: ", symmetry )
   φ = ACE.EuclideanMatrix(Float64,symmetry)
   pibasis = PIBasis(B1p, Bsel; property = φ)
   basis = SymmetricBasis(φ, pibasis)
   @time SymmetricBasis(φ, pibasis)
   @show length(basis)
   BB = evaluate(basis, cfg)

   Iz = findall(iszero, sum(norm, basis.A2Bmap, dims=1)[:])
   if !isempty(Iz)
      @warn("The A2B map for EuclideanMatrix has $(length(Iz))/$(length(basis.pibasis)) zero-columns!!!!")
   end

   ##

   @info("Test FIO")
   using ACEbase.Testing: test_fio
   #println_slim(@test(all(test_fio(basis; warntype = false))))

   ##

   @info("Test equivariance properties for real version")

   tol = 1e-12

   ##
   #                     for (b1, b2) in zip(BB_rot, BB)  
   #print_tf(@test all([ norm(Q' * b1 * Q - b2) < tol
   #                     for (b1, b2) in zip(BB_rot, BB)  ]))
   ##                      
   @info("check for rotation, permutation and inversion equivariance")
   for ntest = 1:30
      local Xs, BB
      try
         Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
      catch
         Xs = [ _randX() for _=1:nX ] 
      end
      BB = evaluate(basis, ACEConfig(Xs))
      Q = rand([-1,1]) * ACE.Random.rand_rot()
      Xs_rot = Ref(Q) .* shuffle(Xs)
      BB_rot = evaluate(basis, ACEConfig(Xs_rot))
      print_tf(@test all([ norm(Q' * b1 * Q - b2) < tol
                           for (b1, b2) in zip(BB_rot, BB)  ]))
   end
   println()

   if symmetry == :general
      @info("Check for some non-symmetric matrix functions")
      for ntest = 1:30
         local Xs, BB
         try
            Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
         catch
            Xs = [ _randX() for _=1:nX ] 
         end
         BB = evaluate(basis, ACEConfig(Xs))
         print_tf(@test any([ b.val != transpose(b.val)
                              for b in BB  ]))
      end
   elseif symmetry == :symmetric
      @info("Check that all matrix functions are symmetric")
      for ntest = 1:30
         local Xs, BB
         try
            Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
         catch
            Xs = [ _randX() for _=1:nX ] 
         end
         BB = evaluate(basis, ACEConfig(Xs))
         print_tf(@test all([ norm(b.val - transpose(b.val))<tol
         for b in BB  ]))
        #print_tf(@test all([ b.val == transpose(b.val)
        #                      for b in BB  ]))
      end
   elseif symmetry == :antisymmetric
      @info("Check that all matrix functions are anti-symmetric")
      for ntest = 1:30
         local Xs, BB
         try
            Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
         catch
            Xs = [ _randX() for _=1:nX ] 
         end
         BB = evaluate(basis, ACEConfig(Xs))
         print_tf(@test all([ b.val == -transpose(b.val)
                              for b in BB  ]))
      end
   end
   println()
   ##

   imtol = 5.0
   @info("Check magnitude of complex part")
   for ntest = 1:30
      local Xs, BB
      try
         Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
      catch
         Xs = [ _randX() for _=1:nX ] 
      end
      BB = evaluate(basis, ACEConfig(Xs))
      for (i,b) in enumerate(BB)
         if norm(imag(b.val)) > imtol
            @warn( "Large imaginary part for $(ACE.get_spec(basis)[i]), $(norm(imag(b.val)))")
         end
      end
      #println(maximum([ norm(imag(b.val))/ norm(real(b.val))  for b in BBs  ]))
      #print_tf(@test all([ norm(imag(b.val)) < .1  for b in BB  ]))
   end
   println()
   print(ACE.get_spec(basis)[1])
end
##

@info("Test equivariance properties for complex version")

basis = SymmetricBasis(φ, pibasis; isreal=false)
# a stupid but necessary test
BB = evaluate(basis, cfg)
BB1 = basis.A2Bmap * evaluate(basis.pibasis, cfg)
println_slim(@test isapprox(BB, BB1, rtol=1e-10)) # MS: This test will fail for isreal=true


# @info("check for rotation, permutation and inversion equivariance")
# for ntest = 1:30
#    local Xs, BB
#    try
#       Xs = rand(PositionState{Float64}, B1p.bases[1], nX)
#    catch
#       Xs = [ _randX() for _=1:nX ] 
#    end
#    BB = evaluate(basis, ACEConfig(Xs))
#    Q = rand([-1,1]) * ACE.Random.rand_rot()
#    Xs_rot = Ref(Q) .* shuffle(Xs)
#    BB_rot = evaluate(basis, ACEConfig(Xs_rot))
#    print_tf(@test all([ norm(Q' * b1 * Q - b2) < tol
#                         for (b1, b2) in zip(BB_rot, BB)  ]))
# end
# println()

# ## keep for further profiling
#
# φ = ACE.EuclideanVector(Complex{Float64})
# pibasis = PIBasis(B1p, ord, maxdeg; property = φ, isreal = false)
# basis = SymmetricBasis(pibasis, φ)
# @time SymmetricBasis(pibasis, φ);
#
# Profile.clear(); # Profile.init(; delay = 0.0001)
# @profile SymmetricBasis(pibasis, φ);
# ProfileView.view()

##

#=
@info(" ... derivatives")
_rrval(x::ACE.XState) = x.rr
for ntest = 1:30
   Us = randn(SVector{3,Float64 }, length(Xs))
   C = randn(typeof(φ.val), length(basis))
   F = t -> sum( sum(c .* b.val)
                 for (c, b) in zip(C, ACE.evaluate(basis, ACEConfig(Xs + t[1] * Us))) )
   dF = t -> [ sum( sum(c .* db)
                    for (c, db) in zip(C, _rrval.(ACE.evaluate_d(basis, ACEConfig(Xs + t[1] * Us))) * Us) ) ]
   print_tf(@test fdtest(F, dF, [0.0], verbose=false))
end
println()
=#

##
