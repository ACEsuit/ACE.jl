module ExportMulti

using YAML
using ACE
using ACE: PIBasis, PIBasisFcn, PIPotential
using ACE.OrthPolys: TransformedPolys
using ACE: rand_radial, cutoff, numz, ZList
using JuLIP: energy, bulk, i2z, z2i, chemical_symbol

function export_ACE(fname, IP)
    #decomposing into V1, V2, V3 (One body, two body and ACE bases)
    V1 = IP.components[1]
    V2 = IP.components[2]
    V3 = IP.components[3]

    species = string.(collect(keys(IP.components[1].E0)))

    species_dict = Dict(zip(collect(0:length(species)-1), species))
    reversed_species_dict = Dict(zip(species, collect(0:length(species)-1)))

    data = Dict()

    if hasproperty(V2, :basis)
        polypairpot = export_polypairpot(V2, reversed_species_dict)
    else hasproperty(V2, :Vin)
        polypairpot = export_polypairpot(V2.Vout, reversed_species_dict)
        reppot = export_reppot(V2, reversed_species_dict)
        data["reppot"] = reppot
    end

    data["polypairpot"] = polypairpot
    #creating "data" dict where we'll store everything


    data["deltaSplineBins"] = 0.001 #" none
    #data["embeddings"] = "none?" ##

    # grabbing the elements key and E0 key from the onebody (V1)
    elements, E0 = export_one_body(V1, species_dict)
    data["elements"] = elements
    data["E0"] = E0

    embeddings, bonds = export_radial_basis(V3, species_dict)
    data["embeddings"] = embeddings
    data["bonds"] = bonds

    functions, lmax = export_ACE_functions(V3, species, reversed_species_dict)
    data["functions"] = functions
    data["lmax"] = lmax

    YAML.write_file(fname, data)
end

function export_one_body(V1, species_dict)
    E0 = []
    elements = []
    for species_ind in keys(species_dict)
        push!(E0, V1(Symbol(species_dict[species_ind])))
        push!(elements, species_dict[species_ind])
    end
    return elements, E0
end

function export_reppot(Vrep, reversed_species_dict)
    reppot = Dict("coefficients" => Dict())

    zlist_dict = Dict(zip(1:length(Vrep.Vout.basis.zlist.list), [string(chemical_symbol(z)) for z in Vrep.Vout.basis.zlist.list]))

    for (index1, element1) in zlist_dict
        for (index2, element2) in zlist_dict
            pair = [reversed_species_dict[element1], reversed_species_dict[element2]]
            coefficients = Dict( "A" => Vrep.Vin[index1, index2].A,
                                "B" => Vrep.Vin[index1, index2].B,
                                "e0" => Vrep.Vin[index1, index2].e0,
                                "ri" => Vrep.Vin[index1, index2].ri) 
            reppot["coefficients"][pair] = coefficients
        end
    end

    return reppot
end

function export_polypairpot(V2, reversed_species_dict)
    Pr = V2.basis.J

    p = Pr.trans.p
    r0 = Pr.trans.r0
    xr = Pr.J.tr
    xl = Pr.J.tl
    pr = Pr.J.pr
    pl = Pr.J.pl
    rcut = cutoff(Pr)
    maxn = length(Pr)

    if length(keys(reversed_species_dict)) == 1
        num_coeffs = length(V2.coeffs)
    else
        num_coeffs = vcat(V2.basis.bidx0...)[2]
    end

    zlist_dict = Dict(zip(1:length(V2.basis.zlist.list), [string(chemical_symbol(z)) for z in V2.basis.zlist.list]))

    polypairpot = Dict( "p" => p,
                        "r0" => r0,
                        "xr" => xr,
                        "xl" => xl,
                        "pr" => pr,
                        "pl" => pl,
                        "rcut" => rcut,
                        "maxn"=> maxn,
                        "recursion_coefficients" => Dict("A" => [Pr.J.A[i] for i in 1:maxn],
                                                         "B" => [Pr.J.B[i] for i in 1:maxn],
                                                         "C" => [Pr.J.C[i] for i in 1:maxn],),
                        "coefficients" => Dict())

    for (index1, element1) in zlist_dict
        for (index2, element2) in zlist_dict
            pair = [reversed_species_dict[element1], reversed_species_dict[element2]]
            ind = V2.basis.bidx0[index1, index2]
            polypairpot["coefficients"][pair] = V2.coeffs[ind+1:ind+num_coeffs]
        end
    end
    
    return polypairpot
end

function export_radial_basis(V3, species_dict)
    #grabbing the transform and basis
    transbasis = V3.pibasis.basis1p.J
    Pr = V3.pibasis.basis1p

    #grabbing all the required params
    p = transbasis.trans.p
    r0 = transbasis.trans.r0
    xr = transbasis.J.tr
    xl = transbasis.J.tl
    pr = transbasis.J.pr
    pl = transbasis.J.pl
    rcut = cutoff(Pr)
    maxn = length(V3.pibasis.basis1p.J.J.A)

    #guessing "radbasname" is that just "polypairpots"
    radbasename = "ACE.jl.base"

    embeddings = Dict()

    for species_ind1 in sort(collect(keys(species_dict)))
        embeddings[species_ind1] = Dict("ndensity" => 1,
                    "FS_parameters" => [1.0, 1.0],
                    "npoti" => "FinnisSinclairShiftedScaled",
                    "drho_core_cutoff" => 1.000000000000000000,
                    "rho_core_cutoff" => 100000.000000000000000000)
    end

    bonds = Dict()
    #this does not respect the coefficient decompositions required per pair
    #need to figure out how to get the right coeffs per pair
    for species_ind1 in sort(collect(keys(species_dict)))
        for species_ind2 in sort(collect(keys(species_dict)))
            pair = [species_ind1, species_ind2]
            bonds[pair] = Dict("p" => p,
                "r0" => r0,
                "xl" => xl,
                "xr" => xr,
                "pr" => pr,
                "pl" => pl,
                "rcut" => rcut,
                "radbasename" => radbasename,
                "maxn" => maxn,
                "recursion_coefficients" => Dict("A" => [Pr.J.J.A[i] for i in 1:maxn],
                                                 "B" => [Pr.J.J.B[i] for i in 1:maxn],
                                                 "C" => [Pr.J.J.C[i] for i in 1:maxn],))
        end
    end

    return embeddings, bonds
end

function export_ACE_functions(V3, species, reversed_species_dict)
    functions = Dict()
    lmax = 0

    for species in values(species)
        sel_bgroups = []
        for i in 1:length(V3.pibasis.inner)
            inner = V3.pibasis.inner[i]
            coeffs = V3.coeffs[i]
            groups = _basis_groups(inner, coeffs)
            for group in groups
                if group["z0"] == ZList(Symbol(species)).list[1]
                    push!(sel_bgroups, Dict("rank" => group["ord"],
                                "mu0" => reversed_species_dict[string(chemical_symbol(group["z0"]))],
                                "ndensity" => 1,
                                "ns" => group["n"],
                                "ls" => group["l"],
                                "mus" => [reversed_species_dict[i] for i in string.(chemical_symbol.(group["zs"]))],
                                "ctildes" => group["C"],
                                "ms_combs" => vcat(group["M"]...),
                                "num_ms_combs" => length(vcat(group["M"]))
                                ))
                    if maximum(group["l"]) > lmax
                        lmax = maximum(group["l"])
                    end
                end
            end
        end
        functions[reversed_species_dict[species]] = sel_bgroups
    end

    return functions, lmax
end

function _basis_groups(inner, coeffs)
    ## grouping the basis functions
    NLZ = []
    M = []
    C = []
    Z0s = []
    for b in keys(inner.b2iAA)
       if coeffs[ inner.b2iAA[b] ] != 0
          push!(Z0s, b.z0)
          push!(NLZ, ( [b1.n for b1 in b.oneps], [b1.l for b1 in b.oneps], [b1.z for b1 in b.oneps]))
          push!(M, [b1.m for b1 in b.oneps])
          push!(C, coeffs[ inner.b2iAA[b] ])
       end
    end
    ords = length.(M)
    perm = sortperm(ords)
    NL = NLZ[perm]
    M = M[perm]
    C = C[perm]
    @assert issorted(length.(M))
    bgrps = []
    alldone = fill(false, length(NL))
    for i = 1:length(NL)
       if alldone[i]; continue; end
       nl = NLZ[i]
       z0 = Z0s[i]
       Inl = findall(NL .== Ref(nl))
       alldone[Inl] .= true
       Mnl = M[Inl]
       Cnl = C[Inl]
       pnl = sortperm(Mnl)
       Mnl = Mnl[pnl]
       Cnl = Cnl[pnl]
       order = length(nl[1])
       push!(bgrps, Dict("n" => nl[1], "l" => nl[2], "z0" => z0, "zs" => nl[3],
                         "M" => Mnl, "C" => Cnl ./ (4*π)^(order/2) , "ord" => order)) #correct?
    end
    return bgrps
end

end


## TO DO
#-  quotations might be an issue, C library allows to export with/without quotes, 
#   Julia does not, maybe we can ask Yury to include them everywhere   
#-  grouping of coefficients per interaction
#-  2B, correct radbasename and required params, as well as maybe repulsive core?
#-  embeddings, if it's linear what do we do?

#export_ACE("test.yace", IP)

# function export_ace_tests(fname::AbstractString, V, ntests = 1;
#                           nrepeat = 1, pert=2.0)
#    at = IPFitting.Data.read_xyz("/Users/Cas/.julia/dev/ACE/src/export/bulk_Al_rep.xyz", energy_key="", force_key="", virial_key="")[1].at * nrepeat
#    JuLIP.set_pbc!(at, false)
#    #r0 = JuLIP.rnn(s)
#    for n = 1:ntests
#       JuLIP.rattle!(at, pert)
#       E = energy(V, at)
#       @show E
#       _write_test(fname * "_$n.dat", JuLIP.positions(at), E)
#    end
#    return at
# end

# function _write_test(fname, X, E)
#    fptr = open(fname; write=true)
#    println(fptr, "E = $E")
#    println(fptr, "natoms = $(length(X))")
#    println(fptr, "# type x y z")
#    for n = 1:length(X)
#       println(fptr, "0 $(X[n][1]) $(X[n][2]) $(X[n][3])")
#    end
#    close(fptr)
# end




#IP.components[2]

#export_ace_tests("/Users/Cas/.julia/dev/ACE/src/export/Ti3Al_rep_1_0", IP)

#export_ACE("./src/export/Ti3Al_rep.yace", IP)


# at = IPFitting.Data.read_xyz("/Users/Cas/.julia/dev/ACE/src/export/bulk.xyz", energy_key="", force_key="", virial_key="")[1].at * 1
# JuLIP.set_pbc!(at, false)
# V1 = energy(IP.components[1], at) 
# V2 = energy(IP.components[2], at) 
# V3 = energy(IP.components[3], at)

# energy(IP, at)

# IP = read_dict(load_dict("./src/export/Ti3Al_basic.json")["IP"])

# Vfit = IP.components[2]

# rp = 2.2
# Vrep = ACE.PairPotentials.RepulsiveCore(Vfit,
#             Dict( ( :Al,  :Al) => (ri = rp, e0 = 0.0),
#                   ( :Ti, :Al) => (ri = rp, e0 =0.0),
#                   ( :Ti, :Ti) => (ri = rp, e0 = 0.0)  ))

# function dimer_energy(IP, r::Float64, spec1, spec2)
#     X = [[0.0,0.0,0.0], [0.0, 0.0, r]]
#     C = [[100.0,0.0,0.0], [0.0, 100.0, 0.0],[0.0, 0.0, 100.0] ]
#     at = Atoms(X, [[0.0,0.0,0.0], [0.0, 0.0, 0.0]], [0.0, 0.0], AtomicNumber.([spec1, spec2]), C, false)
#     return energy(IP, at)
# end

# IP.components[2] = Vrep

# R = [r for r in  1:0.2:5]
# E_TiAl = [(dimer_energy(IP, r, 13, 22)) for r in R]

# for i in 1:21
#     println("$(R[i]) $(E_TiAl[i])")
# end

# data = Dict()

# data["z"] = R
# data["E"] = E_TiAl

# YAML.write_file("/Users/Cas/.julia/dev/ACE/src/export/dimer.yaml", data)

# using Plots

# plot(R, E_TiAl)

# #E_AlAl = [(dimer_energy(IP.components[2], r, 13, 13) ) for r in R]
# #E_AlTi = [(dimer_energy(IP.components[2], r, 13, 22) ) for r in R]


# al = IPFitting.Data.read_xyz("/Users/Cas/.julia/dev/ACE/src/export/bulk_Al_rep.xyz", energy_key="", force_key="", virial_key="")
# at = IPFitting.Data.read_xyz("/Users/Cas/.julia/dev/ACE/src/export/bulk_Al_rep.xyz", energy_key="", force_key="", virial_key="")[1].at 

# JuLIP.rattle!(at, 2.0)
# minimum(IPFitting.Aux.rdf(al, 4.0))

# IP.components[2] = Vrep

# IP.components[2]

# energy(IP, at)

# export_ACE("./src/export/Ti3Al_basic_temp.yace", IP)
# cexport_ace_tests("/Users/Cas/.julia/dev/ACE/src/export/Al_basic_rep", IP)
# #export_ace_tests("/Users/Cas/.julia/dev/ACE/src/export/Ti3Al_test_rep.yace", IP)
# ACE.Export.export_ace("./src/export/Al_basic_rep_old.ace", IP.components[3], IP.components[2], IP.components[1])

# V2.basis
# #grabbing the species and making a dict "0" => "Al", "1" => "Ti"
# #needs to be consistent everywhere!

# export_ACE("./src/export/CHO_pair.yace", IP)

# D1 = YAML.load_file("./src/export/CrFeH_loworder.yace")

# for key in keys(D1["functions"][0][1])
#     @show key, D1["functions"][0][1][key], D2["functions"][0][1][key]
# end

# using IPFitting

# at = IPFitting.Data.read_xyz("/Users/Cas/.julia/dev/ACE/src/export/bulk.xyz", energy_key="", force_key="", virial_key="")

# at = IPFitting.Data.read_xyz("/Users/Cas/.julia/dev/ACE/src/export/bulk.xyz", energy_key="", force_key="", virial_key="")

# at[1].at.Z


# D2["bonds"]


# D2 = YAML.load_file("./src/export/temp2.yace")

# D2["functions"][0][1001]

# 2

# export_polypairpot(V2, species_dict)


# species = string.(collect(keys(IP.components[1].E0)))
# nspecies = length(species)

# #0 index instead of 1 index like Julia

# zlist_dict = Dict(zip(1:length(V2.basis.zlist.list), [string(chemical_symbol(z)) for z in V2.basis.zlist.list]))
# reversed_species_dict = Dict(zip(species, collect(0:length(species)-1)))

# D = Dict()

# num_coeffs = vcat(V2.basis.bidx0...)[2]

# for (index1, element1) in zlist_dict
#     for (index2, element2) in zlist_dict
#         pair = [reversed_species_dict[element1], reversed_species_dict[element2]]
#         ind = V2.basis.bidx0[index1, index2]
#         D[pair] = V2.coeffs[ind+1:ind+num_coeffs]
#     end
# end


# D

# V2.basis.bidx0[2,1]

# V2.basis.  

# V2.coeffs

# V3.basis
# ###########################################

# al_train = IPFitting.Data.read_xyz(@__DIR__() * "/train50.xyz", energy_key="energy", force_key="forces");

# #####################
# #SETTING UP BASIS

# train_size = 50

# r0 = 0.8
# r0_2b = 1
# N = 3
# deg_site = 6
# deg_pair = 3

# 5.5 * 0.8

# # construction of a basic basis for site energies
# Bsite = rpi_basis(species = [:C, :H, :O],
#                   N = N,       # correlation order = body-order - 1
#                   maxdeg = deg_site,  # polynomial degree
#                   r0 = r0,     # estimate for NN distance
#                   rin = 0.8*r0, rcut = 5.5*r0,   # domain for radial basis (cf documentation)
#                   pin = 2)                     # require smooth inner cutoff

# # pair potential basis
# Bpair = pair_basis(species = [:C, :H, :O], r0 = r0_2b, maxdeg = deg_pair,
#                    rcut = 7.0 * r0_2b, rin = 0.0,
#                    pin = 0 )   # pin = 0 means no inner cutoff


# B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

# @show length(B)
# dB = LsqDB("", B, al_train)

# ##############
# #GET LINEAR SYSTEM

# weights = Dict("default" => Dict("E" => 15.0, "F" => 1.0 , "V" => 1.0 ))

# E0_O = -2035.5709809589698
# E0_H = -13.568422383046626
# E0_C = -1025.2770951782686

# Vref = OneBody(:C => E0_C, :H => E0_H, :O => E0_O)

# IP, lsqinfo = IPFitting.Lsq.lsqfit(dB, Vref=Vref,
#                      solver=(:rid, 1.2),
#                      asmerrs=true, weights=weights)

# rmse_table(lsqinfo["errors"])

# save_dict(@__DIR__() * "/CHO_test.json", Dict("IP" => write_dict(IP)))


# ###

# #D1["functions"]


# #D2["functions"]["Al"]


# #species = string.(collect(keys(IP.components[1].E0)))
# # nspecies = length(species)

# # elements, E0 = export_ACE_V1(V1, species_dict)

# # #0 index instead of 1 index like Julia
# # species_dict = Dict(zip(collect(0:length(species)-1), species))

# # data = Dict()
# # #creating "data" dict where we'll store everything

# # embeddings, bonds = export_ACE_V2(V2, species_dict)

# # data["embeddings"] = embeddings
# # data["bonds"] = bonds

# # functions = Dict()


# # species_dict = Dict(zip(collect(0:length(species)-1), species))
# #     reversed_species_dict = Dict(zip(species, collect(0:length(species)-1)))
# # functions = export_ACE_V3(V3, reversed_species_dict)

# # functions

# # data["functions"] = functions

# # YAML.write_file("./src/export/temp.yace", data)

# data = Dict

# data["E0"] = ""

# vcat(groups[100]["M"])

# inner = V3.pibasis.inner[2]
# coeffs = V3.coeffs[2]

#     #grabbing the basis functions, without interation specified
# groups = _basis_groups(inner, coeffs)


# groups[1]

# reversed_species_dict = Dict(zip(species, collect(0:length(species)-1)))

# (groups[1]["zs"][1])

# reversed_species_dict

# [reversed_species_dict[i] for i in string.(chemical_symbol.(groups[1]["zs"]))]

# species_dict

# reversed_species_dict = map(reverse, species_dict)

# #function export_functions(V3)
# inner = V3.pibasis.inner[end]
# coeffs = V3.coeffs[1]

# #grabbing the basis functions, without interation specified
# groups = _basis_groups(inner, coeffs)

# #a = Dict("groups" => 


# groups[100]["C"]
# #groups[100]


# inner.z0
#ls 
