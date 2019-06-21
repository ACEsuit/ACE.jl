
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs
using Test, Printf, LinearAlgebra, StaticArrays, BenchmarkTools, Test
using SHIPs: generate_LK, generate_LK_tuples
using SHIPs.SphericalHarmonics: ClebschGordan, cg1

printred(s) = printstyled(s, bold=true, color=:red)
print_tf(::Test.Pass) = printstyled("+", bold=true, color=:green)
print_tf(::Test.Fail) = printstyled("-", bold=true, color=:red)

function naive_Bcoeff(ll::SVector{4}, mm, cg)
   coeff = 0.0
   @assert sum(mm) ≈ 0
   M = mm[1] + mm[2]
   for J = max(abs(ll[1]-ll[2]), abs(ll[3]-ll[4])):min(ll[1]+ll[2], ll[3]+ll[4])
      # @assert cg1(ll[1], mm[1], ll[2], mm[2], J, M) == cg(ll[1], mm[1], ll[2], mm[2], J, M)
      # @assert cg1(ll[3], mm[3], ll[4], mm[4], J, -M) == cg(ll[3], mm[3], ll[4], mm[4], J, -M)
      coeff += ( (-1)^(M) *
                  cg1(ll[1], mm[1], ll[2], mm[2], J, M) *
                  cg1(ll[3], mm[3], ll[4], mm[4], J, -M) )
   end
   return coeff
end

function naive_Bcoeff(ll::SVector{3}, mm, cg)
   @assert ( cg1(ll[1], mm[1], ll[2], mm[2], ll[3], -mm[3]) ≈
              cg(ll[1], mm[1], ll[2], mm[2], ll[3], -mm[3]) )
   return (-1)^mm[3] * cg(ll[1], mm[1], ll[2], mm[2], ll[3], -mm[3])
end

function check_Bcoeffs(ll::SVector, cg)
   allzeros = true
   for mpre in SHIPs._mrange(ll)
      mend = - sum(Tuple(mpre))
      if abs(mend) > ll[end]; continue; end
      mm = SVector(Tuple(mpre)..., mend)
      C = naive_Bcoeff(ll, mm, cg)
      _C = SHIPs._Bcoeff(ll, mm, cg)
      if !( (C ≈ _C) || (abs(C-_C) < 1e-12) )
         return false, allzeros
      end
      if !(C ≈ 0);  allzeros = false; end
   end
   return true, allzeros
end

##
@info("Testing _mrange")
ll = SVector(2,3,0)
mrange2 = SHIPs._mrange(ll)
println(@test mrange2 == CartesianIndices( (-2:2, -3:3) ))
ll = SVector(4,2,5,0)
mrange3 = SHIPs._mrange(ll)
println(@test mrange3 == CartesianIndices( (-4:4, -2:2, -5:5) ))

##
deg, wY = 5, 1.0
cg = ClebschGordan(deg)
KL, Nu =  generate_LK_tuples(deg, wY, 3, cg; filter=false)
Nu_filter = SHIPs.filter_tuples(KL, Nu, Val(3), cg)
Izero = Int[]
Iodd = Int[]
@info("Testing the RI coefficients for deg = $deg, 4B")
for (i, ν) in enumerate(Nu)
   global Izero, Iodd
   ll = SVector([KL[ν[i]].l for i = 1:length(ν)]...)
   pass, isz = check_Bcoeffs(ll, cg)
   if isz; push!(Izero, i); end
   if isodd(sum(ll)); push!(Iodd, i); end
   print_tf(@test pass)
end
println()
Izodd = union(Izero, Iodd)
@info("""   #Nu = $(length(Nu)), #Nu_filt = $(length(Nu_filter)), #(zero or odd) = $(length(Izodd)) """)
println(@test (length(Nu) == length(Nu_filter) + length(Izodd)))


##
deg, wY = 10, 2.0
cg = ClebschGordan(deg)
KL, Nu =  generate_LK_tuples(deg, wY, 4, cg; filter=false)
Nu_filter = SHIPs.filter_tuples(KL, Nu, Val(4), cg)
Izero = Int[]
Iodd = Int[]
@info("Testing the RI coefficients for deg = $deg, 5B")
for (i, ν) in enumerate(Nu)
   global Izero, Iodd
   ll = SVector([KL[ν[i]].l for i = 1:length(ν)]...)
   pass, isz = check_Bcoeffs(ll, cg)
   if isz; push!(Izero, i); end
   if isodd(sum(ll)); push!(Iodd, i); end
   print_tf(@test pass)
end
println()
Izodd = union(Izero, Iodd)
@info("""   #Nu = $(length(Nu)), #Nu_filt = $(length(Nu_filter)), #(zero or odd) = $(length(Izodd)) """)
println(@test (length(Nu) == length(Nu_filter) + length(Izodd)))
