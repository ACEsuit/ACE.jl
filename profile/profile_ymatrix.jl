##

using ACE
using StaticArrays, Random, Printf, Test, LinearAlgebra, ACE.Testing
using ACE: evaluate, evaluate_d, SymmetricBasis, PIBasis, O3, rand_vec3 
using ACE.Random: rand_rot, rand_refl
using ACEbase.Testing: fdtest, println_slim
using ACE.Testing: __TestSVec

# using Profile, ProfileView

# Extra using Wigner for computing Wigner Matrix
using ACE.Wigner
using ACE.Wigner: get_orbsym

##
# construct the 1p-basis
maxdeg = 6
ord = 3
Bsel = SimpleSparseBasis(ord, maxdeg)

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)

# generate a configuration
nX = 10

_randX() = State(rr = rand_vec3(B1p["Rn"]))
Xs = [ _randX() for _=1:nX ]
cfg = ACEConfig(Xs)


## Keep for futher profiling

maxdeg = 10
ord = 2
Bsel = SimpleSparseBasis(ord, maxdeg)

φ = ACE.SphericalMatrix(2,2; T = ComplexF64)
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
basis = SymmetricBasis(φ, B1p, Bsel)
@show length(basis)

Xs = [ _randX() for _=1:200 ]

##

using BenchmarkTools


# @btime evaluate($basis, $cfg)

# @profview begin
#    for _ = 1:10_000   
#       evaluate(basis, cfg)
#    end
# end

##

c = randn(length(basis)) ./ (1:length(basis)).^2
m = ACE.LinearACEModel(basis, c)

# @btime evaluate($m, $cfg)

# @profview begin
#    for _ = 1:40_000
#       evaluate(m, cfg)
#    end
# end

## 

AA = evaluate(basis.pibasis, cfg)
C̃ = m.evaluator.coeffs
C = [ c.val for c in C̃ ]
C1 = randn(length(AA))


using LoopVectorization

##

_contract1(AA, C) = sum( AA[i] * C[i] for i = 1:length(AA) )

function _contract2(AA, C)
   @assert length(AA) == length(C)
   val = AA[1] * C[1] 
   @inbounds for i = 2:length(AA)
      val += AA[i] * C[i]
   end
   return val 
end

function _contract3(AA, C)
   @assert length(AA) == length(C)
   val = AA[1] * C[1] 
   for i = 2:length(AA)
      @fastmath @inbounds val += AA[i] * C[i]
   end
   return val 
end



##

@info("With Matrix")
@btime _contract1($AA, $C)
@btime _contract2($AA, $C)
@btime _contract3($AA, $C)

@info("With Scalar")
@btime _contract1($AA, $C1)
@btime _contract2($AA, $C1)
@btime _contract3($AA, $C1)


##

C = randn(296, length(AA))
C * AA

@btime $C * $AA 

AAA = zeros(eltype(AA), length(AA), 512)
for i = 1:512
   AAA[:, i] = AA
end

@btime $C * $AAA

AAA1 = collect(AAA')
BB = AAA1 * C1
@btime mul!($BB, $AAA1, $C1)
@btime dot($AA, $C1)

C10 = rand(4089, 16)
BB10 = AAA1 * C10 
@btime mul!($BB10, $AAA1, $C10)