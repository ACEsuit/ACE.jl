
# A demonstration how to use the new SparsePSHDegreeM degree type
# see also ?ACE.RPI.SparsePSHDegreeM

using ACE, JuLIP
using ACE: z2i, i2z, order
#---

zTi = AtomicNumber(:Ti)
zAl = AtomicNumber(:Al)

# the weights for the n
Dn = Dict( "default" => 1.0,
            (zTi, zAl) => 2.0,    # weak interaction between species
            (zAl, zTi) => 2.0,    # weak interaction between species
            (zAl, zAl) => 1.2,    # slightly smaller basis for Al than for Ti
            (2, zTi, zAl) => 0.8  # but for 3-body we override this ...
         )

# the weights for the l
Dl = Dict( "default" => 1.5, )    # let's do nothing special here...
                                  # but same options are available for for n

# the degrees
Dd = Dict( "default" => 10,
           1 => 20,     # N = 1
           2 => 20,     # ...
           3 => 15,
           (2, zTi) => 25   # an extra push for the 3-body Ti basis
        )                   # (probably a dumb idea, just for illustration!)

Deg = ACE.RPI.SparsePSHDegreeM(Dn, Dl, Dd)

#---

# generate basis
# - note that degree is already incorporated into Deg
#   but we can still enlarge it e.g. by using maxdeg = 1.2, 1.5, 2.0, ...
basis = ACE.Utils.rpi_basis(species = [:Ti, :Al],
                              N = 5,
                              r0 = rnn(:Ti),
                              D = Deg,
                              maxdeg = 1)

#---
# analyse the basis a bit to see the effect this had

iAl = z2i(basis, AtomicNumber(:Al))
iTi = z2i(basis, AtomicNumber(:Ti))

println("The Ti site has more basis functions than the Al site")
@show length(basis.pibasis.inner[iAl])
@show length(basis.pibasis.inner[iTi])

specTi = collect(keys(basis.pibasis.inner[iTi].b2iAA))
specTi2 = specTi[ order.(specTi) .== 2 ]
specTi3 = specTi[ order.(specTi) .== 3 ]

println("The Ti-2N interaction has more basis functions than Ti-3N interaction")
@show length(specTi2)
@show length(specTi3)

specTi2_Ti = specTi2[ [ all(b.z == zTi for b in B.oneps) for B in specTi2 ] ]
specTi2_Al = specTi2[ [ all(b.z == zAl for b in B.oneps) for B in specTi2 ] ]

println("The Ti-Ti interaction has more basis functions than Ti-Al interaction")
@show length(specTi2_Ti)
@show length(specTi2_Al)
