using YAML
using ACE
using ACE: PIBasis, PIBasisFcn, PIPotential
using ACE.OrthPolys: TransformedPolys
using ACE: rand_radial, cutoff, numz
using JuLIP: energy, bulk, i2z, z2i, chemical_symbol

function _basis_groups(inner, coeffs)
    ## grouping the basis functions
    NL = []
    M = []
    C = []
    for b in keys(inner.b2iAA)
       if coeffs[ inner.b2iAA[b] ] != 0
          push!(NL, ( [b1.n for b1 in b.oneps], [b1.l for b1 in b.oneps] ))
          push!(M, [b1.m for b1 in b.oneps])
          push!(C, coeffs[ inner.b2iAA[b] ])
       end
    end
    ords = length.(M)
    perm = sortperm(ords)
    NL = NL[perm]
    M = M[perm]
    C = C[perm]
    @assert issorted(length.(M))
    bgrps = []
    alldone = fill(false, length(NL))
    for i = 1:length(NL)
       if alldone[i]; continue; end
       nl = NL[i]
       Inl = findall(NL .== Ref(nl))
       alldone[Inl] .= true
       Mnl = M[Inl]
       Cnl = C[Inl]
       pnl = sortperm(Mnl)
       Mnl = Mnl[pnl]
       Cnl = Cnl[pnl]
       push!(bgrps, Dict("n" => nl[1], "l" => nl[2],
                         "M" => Mnl, "C" => Cnl))
    end
    return bgrps
end

function _write_group(g; ndensitymax = 1)
    ## write functions to function dict in YACE format
    order = length(g["l"])
    mu0 = 0
    #mu = order
    num_ms = length(g["M"])
    ns = g["n"]
    ls = g["l"]
    c_tildes = (g["C"] ./ (4*π)^(order/2))
    return "{mu0: $(mu0), rank: $(order), ndensitymax: $(ndensitymax), num_ms_combs: $(num_ms), mus!!, ns: $(ns), ls: $(ls), ms_combs!!!, c_tildes: $(c_tildes))}"
end


function export_ACE(fname, IP)
    #decomposing into V1, V2, V3 (One body, two body and ACE bases)
    V1 = IP.components[1]
    V2 = IP.components[2]
    V3 = IP.components[3]

    #grabbing the species and making a dict "0" => "Al", "1" => "Ti"
    #needs to be consistent everywhere!
    species = string.(collect(keys(IP.components[1].E0)))
    nspecies = length(species)

    #0 index instead of 1 index like Julia
    species_dict = Dict(zip(collect(0:length(species)-1), species))

    #creating "data" dict where we'll store everything
    data = Dict()

    # not sure about both keys below... ask Yury?
    # both are not in the previous verison of PACE
    data["deltaSplineBins"] = "0.001" 
    data["embeddings"] = "none?" ##

    # grabbing the elements key and E0 key from the onebody (V1)
    elements, E0 = export_ACE_V1(V1, species_dict)
    data["elements"] = elements
    data["E0"] = E0

    #storing all info with PACE
    bonds = export_ACE_V2(V2, species_dict)
    data["bonds"] = bonds

    functions = export_ACE_V3(V3, species_dict)
    data["functions"] = functions
    ## 1, 2 indices for multispecies support 

    YAML.write_file(fname, data)
end

function YACE_format(l)
    return "[" * join(l, ",") * "]"
end

function export_ACE_V1(V1, species_dict)
    E0 = []
    elements = []
    for species_ind in keys(species_dict)
        push!(E0, V1(Symbol(species_dict[species_ind])))
        push!(elements, species_dict[species_ind])
    end
    return YACE_format(elements), YACE_format(E0)
end

function export_ACE_V2(V2, species_dict)
    #these are ALL coeffs, but these are species dependent [0,0], [1,1]
    #we need to decompose coeffs and assign them in the right bond key
    #I'm not sure how..
    coeffs = V2.coeffs

    #grabbing the transform and basis
    transbasis = V2.basis.J
    Pr = V2.basis

    #grabbing all the required params
    p = transbasis.trans.p
    r0 = transbasis.trans.r0
    xr = transbasis.J.tr
    xl = transbasis.J.tl
    pr = transbasis.J.pr
    pl = transbasis.J.pl
    rcut = cutoff(Pr)
    maxn = length(Pr)

    #guessing "radbasname" is that just "polypairpots"
    radbasename = "polypairpots"

    bonds = Dict()
    #this does not respect the coefficient decompositions required per pair
    #need to figure out how to get the right coeffs per pair
    for species_ind1 in sort(collect(keys(species_dict)))
        for species_ind2 in sort(collect(keys(species_dict)))
            bonds["[$(species_ind1), $(species_ind2)]"] = "{p: $(p), r0: $(r0), xr: $(xr), xl: $(xl), pr: $(pr), pl: $(pl), rcut: $(rcut), radbasename: $(radbasename), maxn: $(maxn), coefficients: $(coeffs)}"
        end
    end

    return bonds
end

function export_ACE_V3(V3, species_dict)
    #we'll need to species dict here to also sort the coeffs per interation
    inner = V3.pibasis.inner[1]
    coeffs = V3.coeffs[1]

    #grabbing the basis functions, without interation specified
    groups = _basis_groups(inner, coeffs)

    #dumping all coeffs under "0", this is wrong
    #functions needs to respect the species_dict
    functions = Dict("0" => [])
    for group in groups
        push!(functions["0"],_write_group(group))
    end
    #HOW DO WE GET NSPECIES?
    return functions
end

## TO DO
#-  quotations might be an issue, C library allows to export with/without quotes, 
#   Julia does not, maybe we can ask Yury to include them everywhere   
#-  grouping of coefficients per interaction
#-  2B, correct radbasename and required params, as well as maybe repulsive core?
#-  embeddings, if it's linear what do we do?

export_ACE("test.yace", IP)
