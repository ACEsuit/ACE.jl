
@testset "B-coefficients" begin

using PoSH
using Test, Printf, LinearAlgebra, StaticArrays, BenchmarkTools, Test
using PoSH: generate_ZKL, generate_ZKL_tuples, SparseSHIP, maxL
using PoSH.Rotations: ClebschGordan, clebschgordan
using JuLIP.Testing


# function naive_Bcoeff(ll::SVector{4}, mm, cg)
#    coeff = 0.0
#    @assert sum(mm) ≈ 0
#    M = mm[1] + mm[2]
#    for J = max(abs(ll[1]-ll[2]), abs(ll[3]-ll[4])):min(ll[1]+ll[2], ll[3]+ll[4])
#       # @assert clebschgordan(ll[1], mm[1], ll[2], mm[2], J, M) == cg(ll[1], mm[1], ll[2], mm[2], J, M)
#       # @assert clebschgordan(ll[3], mm[3], ll[4], mm[4], J, -M) == cg(ll[3], mm[3], ll[4], mm[4], J, -M)
#       coeff += ( (-1)^(M) *
#                   clebschgordan(ll[1], mm[1], ll[2], mm[2], J, M) *
#                   clebschgordan(ll[3], mm[3], ll[4], mm[4], J, -M) )
#    end
#    return coeff
# end
#
# function naive_Bcoeff(ll::SVector{3}, mm, cg)
#    @assert ( clebschgordan(ll[1], mm[1], ll[2], mm[2], ll[3], -mm[3]) ≈
#               cg(ll[1], mm[1], ll[2], mm[2], ll[3], -mm[3]) )
#    return (-1)^mm[3] * cg(ll[1], mm[1], ll[2], mm[2], ll[3], -mm[3])
# end
#
# function check_Bcoeffs(ll::SVector, cg)
#    allzeros = true
#    for mm in PoSH._mrange(ll)
#       if abs(mm[end]) > ll[end]; continue; end
#       C = naive_Bcoeff(ll, mm, cg)
#       _C = PoSH._Bcoeff(ll, mm, cg)
#       if !( (C ≈ _C) || (abs(C-_C) < 1e-12) )
#          return false, allzeros
#       end
#       if !(C ≈ 0);  allzeros = false; end
#    end
#    return true, allzeros
# end

##
@info("Testing _mrange")
ll = SVector(2,3,4)
mrange2 = collect(PoSH._mrange(ll))
testrg = CartesianIndices( (-2:2, -3:3) )
println(@test all( mpre -> (abs(sum(mpre.I)) > ll[end]) ||
                           (SVector(mpre.I..., -sum(mpre.I)) ∈ mrange2),
                   testrg ))

ll = SVector(4,2,3,4)
mrange3 = collect(PoSH._mrange(ll))
testrg = CartesianIndices( (-4:4, -2:2, -3:3) )
println(@test all( mpre -> (abs(sum(mpre.I)) > ll[end]) ||
                           (SVector(mpre.I..., -sum(mpre.I)) ∈ mrange3),
                   testrg ))

##
Deg = SparseSHIP(3, 5; wL = 1.0)
Bcoefs = PoSH.Rotations.CoeffArray()
KL, Nu =  generate_ZKL_tuples(Deg, Bcoefs; filter=false)
KL = KL[1]
_, Nu_filter = generate_ZKL_tuples(Deg, Bcoefs; filter=true)
Izero = Int[]
Iodd = Int[]

Nu3 = Nu[3]
Nu3_filter = Nu_filter[3]

@info("Testing the RI coefficients for Deg = $Deg, 4B")
for (i, zν) in enumerate(Nu3)
   # global Izero, Iodd
   ν = zν.ν
   ll = getfield.(KL[ν], :l)    # SVector([KL[ν[i]].l for i = 1:length(ν)]...)
   # pass, isz = check_Bcoeffs(ll, cg)
   # if isz; push!(Izero, i); end
   Clm = PoSH.Rotations.single_B(Bcoefs, ll)
   if norm(Clm) == 0; push!(Izero, i); end
   if isodd(sum(ll)); push!(Iodd, i); end
   # print_tf(@test pass)
end
println()
Izodd = union(Izero, Iodd)
@info("""   #Nu = $(length(Nu3)), #Nu_filt = $(length(Nu3_filter)), #(zero or odd) = $(length(Izodd)) """)
println(@test (length(Nu3) == length(Nu3_filter) + length(Izodd)))

##
Deg = SparseSHIP(4, 10; wL = 2.0)
Bcoefs = PoSH.Rotations.CoeffArray()
KL, Nu =  generate_ZKL_tuples(Deg, Bcoefs; filter=false)
KL = KL[1]
_, Nu_filter = generate_ZKL_tuples(Deg, Bcoefs; filter=true)
Nu4 = Nu[4]
Nu4_filter = Nu_filter[4]
Izero = Int[]
Iodd = Int[]
@info("Testing the RI coefficients for Deg = $Deg, 5B")
for (i, zν) in enumerate(Nu4)
   # global Izero, Iodd
   ν = zν.ν
   ll = SVector([KL[ν[i]].l for i = 1:length(ν)]...)
   # pass, isz = check_Bcoeffs(ll, cg)
   Clm = PoSH.Rotations.single_B(Bcoefs, ll)
   isz = (norm(Clm) == 0)
   if isz; push!(Izero, i); end
   if isodd(sum(ll)); push!(Iodd, i); end
   # print_tf(@test pass)
end
println()
Izodd = union(Izero, Iodd)
@info("""   #Nu = $(length(Nu4)), #Nu_filt = $(length(Nu4_filter)), #(zero or odd) = $(length(Izodd)) """)
println(@test (length(Nu4) == length(Nu4_filter) + length(Izodd)))


end
