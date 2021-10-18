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

    if hasproperty(V2, :basis)
        species = collect(string.(chemical_symbol.(V2.basis.zlist.list.data)))
    else hasproperty(V2, :Vout)
        species = collect(string.(chemical_symbol.(V2.Vout.basis.zlist.list.data)))
    end

    species_dict = Dict(zip(collect(0:length(species)-1), species))
    reversed_species_dict = Dict(zip(species, collect(0:length(species)-1)))

    data = Dict()

    data["deltaSplineBins"] = 0.001 #" none

    elements = Vector(undef, length(species))
    E0 = zeros(3)
    
    for (index, element) in species_dict
        E0[index+1] = V1(Symbol(element))
        elements[index+1] = element
    end

    # grabbing the elements key and E0 key from the onebody (V1)
    data["elements"] = elements
    data["E0"] = E0

    if hasproperty(V2, :basis)
        polypairpot = export_polypairpot(V2, reversed_species_dict)
    else hasproperty(V2, :Vin)
        polypairpot = export_polypairpot(V2.Vout, reversed_species_dict)
        reppot = export_reppot(V2, reversed_species_dict)
        data["reppot"] = reppot
    end

    data["polypairpot"] = polypairpot
    #creating "data" dict where we'll store everything

    embeddings, bonds = export_radial_basis(V3, species_dict)
    data["embeddings"] = embeddings
    data["bonds"] = bonds

    functions, lmax = export_ACE_functions(V3, species, reversed_species_dict)
    data["functions"] = functions
    data["lmax"] = lmax

    YAML.write_file(fname, data)
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

    for i in 1:length(V3.pibasis.inner)
        sel_bgroups = []
        inner = V3.pibasis.inner[i]
        z0 = V3.pibasis.inner[i].z0
        coeffs = V3.coeffs[i]
        groups = _basis_groups(inner, coeffs)
        for group in groups
            for (m, c) in zip(group["M"], group["C"])
                c_ace = c / (4*π)^(group["ord"]/2)
                #@show length(c_ace)
                ndensity = 1
                push!(sel_bgroups, Dict("rank" => group["ord"],
                            "mu0" => reversed_species_dict[string(chemical_symbol(group["z0"]))],
                            "ndensity" => ndensity,
                            "ns" => group["n"],
                            "ls" => group["l"],
                            "mus" => [reversed_species_dict[i] for i in string.(chemical_symbol.(group["zs"]))],
                            "ctildes" => [c_ace],
                            "ms_combs" => m,
                            "num_ms_combs" => length([c_ace])))
                if maximum(group["l"]) > lmax
                    lmax = maximum(group["l"])
                end
            end
        end
        functions[reversed_species_dict[string(chemical_symbol(z0))]] = sel_bgroups
    end

    return functions, lmax
end

function _basis_groups(inner, coeffs)
    ## grouping the basis functions
    NLZZ = []
    M = []
    C = []
    for b in keys(inner.b2iAA)
       if coeffs[ inner.b2iAA[b] ] != 0
          push!(NLZZ, ( [b1.n for b1 in b.oneps], [b1.l for b1 in b.oneps], [b1.z for b1 in b.oneps], b.z0))
          push!(M, [b1.m for b1 in b.oneps])
          push!(C, coeffs[ inner.b2iAA[b] ])
       end
    end
    ords = length.(M)
    perm = sortperm(ords)
    NLZZ = NLZZ[perm]
    M = M[perm]
    C = C[perm]
    @assert issorted(length.(M))
    bgrps = []
    alldone = fill(false, length(NLZZ))
    for i = 1:length(NLZZ)
       if alldone[i]; continue; end
       nlzz = NLZZ[i]
       Inl = findall(NLZZ .== Ref(nlzz))
       alldone[Inl] .= true
       Mnl = M[Inl]
       Cnl = C[Inl]
       pnl = sortperm(Mnl)
       Mnl = Mnl[pnl]
       Cnl = Cnl[pnl]
       order = length(nlzz[1])
       push!(bgrps, Dict("n" => nlzz[1], "l" => nlzz[2], "z0" => nlzz[4], "zs" => nlzz[3],
                         "M" => Mnl, "C" => Cnl, "ord" => order)) #correct?
    end
    return bgrps
end

end