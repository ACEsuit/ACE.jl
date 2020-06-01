
 using SHIPs, JuLIP, Test, JuLIP.Testing
 using JuLIP: evaluate, evaluate_d
 using SHIPs: PIBasisFcn
 using SHIPs.RPI: BasicPSH1pBasis, PSH1pBasisFcn

 D = JuLIP.load_dict(@__DIR__() * "/ship_v5.json")

 ##

 test = D["tests"]

 species = D["zlist"]["list"]
 zlist = read_dict(D["zlist"])
 rawcoeffs = Float64.(D["coeffs_re"][1])

 ##
 # get the radial basis

 function import_rbasis_v05(D, rtests = [])
    @assert D["__id__"] == "SHIPs_TransformedJacobi"
    trans = SHIPs.Transforms.PolyTransform(D["trans"])
    ru, rl = D["ru"], D["rl"]
    tu, tl = trans(ru), trans(rl)
    pl = pu = D["cutoff"]["P"]
    if tl > tu
       tl, tr = tu, tl
       pl, pr = pu, pl
    else
       tl, tr = tl, tu
       pl, pr = pl, pu
    end

    A = Float64.(D["jacobicoeffs"]["A"])
    B = Float64.(D["jacobicoeffs"]["B"])
    C = Float64.(D["jacobicoeffs"]["C"])
    nrm = Float64.(D["jacobicoeffs"]["nrm"])
    deg = D["deg"]
    α, β = D["a"], D["b"]

    An = zeros(length(A)+1)
    Bn = zeros(length(An))
    Cn = zeros(length(An))
    An[1] = nrm[1]
    An[2] = - nrm[2]/nrm[1] * 0.5 * (α+β+2)
    Bn[2] = nrm[2]/nrm[1] * ((α+1) - 0.5 * (α+β+2))
    for n = 3:length(An)
       An[n] = - nrm[n] / nrm[n-1] * A[n-1]
       Bn[n] = nrm[n] / nrm[n-1] * B[n-1]
       Cn[n] = nrm[n] / nrm[n-2] * C[n-1]
    end
    # return SHIPs.OrthPolys.OrthPolyBasis(2, -1.0, 2, 1.0, An, Bn, Cn,
    #                                      Float64[], Float64[])

    # transform to the correct interval
    a = (2/(tr-tl))
    At = An * a
    Bt = Bn - An * ((tr+tl) / (tr-tl))
    Ct = Cn
    At[1] = An[1] * a^4

    Jt = SHIPs.OrthPolys.OrthPolyBasis(pl, tl, pr, tr, At, Bt, Ct,
                                      Float64[], Float64[])
    Jr = SHIPs.OrthPolys.TransformedPolys(Jt, trans, rl, ru)

    if !isempty(rtests) > 0
       @info("Running consistency tests:")
       r = rtests[1]["r"]
       Jr_test = rtests[1]["Pr"]
       Jr_new = evaluate(Jr, r)
       f = Jr_test[1] / Jr_new[1]
       Jr.J.A[1] *= f
    end
    for n = 1:length(rtests)
       r = rtests[1]["r"]
       Jr_test = rtests[1]["Pr"]
       Jr_new = evaluate(Jr, r)
       print_tf(@test Jr_test ≈ Jr_new)
    end
    return Jr
 end

 Pr = import_rbasis_v05(D["J"], D["rtests"])


 ##
 # get the 1-p and n-p basis specifications

 rawspec = D["aalists"][1]["ZKLM_list"]
 pispec = []
 pispec_i = []
 spec1 = []
 coeffs = []
 for (idx, ZNLM) in enumerate(rawspec)
    izs = ZNLM[1]
    ns = ZNLM[2] .+ 1
    ls = ZNLM[3]
    ms = ZNLM[4]
    order = length(ns)
    zs = [AtomicNumber(species[iz]) for iz in izs]
    bs = PSH1pBasisFcn[]
    bs_i = Int[]
    for α = 1:order
       b = PSH1pBasisFcn(ns[α], ls[α], ms[α], zs[α])
       I1 = findall(isequal(b), spec1)
       if isempty(I1)
          push!(spec1, b)
          i1 = length(spec1)
       else
          @assert length(I1) == 1
          i1 = I1[1]
       end
       push!(bs, b)
       push!(bs_i, i1)
    end
    p = sortperm(bs_i)
    bs_i = bs_i[p]
    bb = PIBasisFcn(AtomicNumber(species[1]), bs[p])
    In = findall(isequal(bs_i), pispec_i)
    if isempty(In)
       push!(pispec, bb)
       push!(pispec_i, bs_i)
       push!(coeffs, rawcoeffs[idx])
    else
       @assert length(In) == 1
       in = In[1]
       coeffs[in] += rawcoeffs[idx]
    end
 end

 @show length(spec1)
 @show length(pispec)
 ;
 ##

 basis1p = BasicPSH1pBasis(Pr, zlist, identity.(spec1))
 AAindices = 1:length(pispec)

 inner = SHIPs.InnerPIBasis(spec1, pispec, AAindices, zlist.list[1])
 pibasis = PIBasis(basis1p, zlist, (inner,))

 # fix permutation of coefficients?
 perm = [ inner.b2iAA[b]  for b in pispec ]
 invperm = sortperm(perm)
 V = PIPotential(pibasis, (Float64.(coeffs[invperm]),))


 tests = D["tests"]
 for t1 in tests
    Rs = JVecF.(t1["Rs"])
    Zs = AtomicNumber.(t1["Zs"])
    z0 = AtomicNumber(t1["z0"])
    valold = t1["val"]
    valnew = evaluate(V, Rs, Zs, z0)
    @show valold, valnew
 end
