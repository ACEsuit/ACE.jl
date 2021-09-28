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