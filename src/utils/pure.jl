"""
`module Pure`

This module contains utility functions to obtain the contribution of basis functions of each body-order. 
"""
module Pure

include("../extimports.jl")
include("../aceimports.jl")

using JuLIP

export nB_energy, nB_potential


"""
This function returns the Body ordered contributions to the total ACE energy
  Arguments:
    IP: PIPotential object
    atoms: Atoms object whose energy should be returned
    B_order: Int, specifying the body order (correlation order + 1)
  Returns: 
    Contribution of the N-Body basis functions to the energy. 
"""
function nB_energy(IP::PIPotential, atoms::Atoms, B_order::Int)
  IP_new = nB_potential(IP, B_order)
  return energy(IP_new, atoms)
end


"""
This function returns the n-Body part of the ACE basis set
  Arguments:
    IP: PIPotential object
    atoms: Atoms object whose energy should be returned
    B_order: Int, specifying the body order (correlation order + 1)
  Returns: 
    Potential consisting of the N-Body basis functions only. 
"""
function nB_potential(IP::PIPotential, B_order::Int)
  korr_ord = B_order - 1
  Cs = deepcopy(IP.coeffs)
  for i = 1:numz(IP), j = 1:length(Cs[i])
    ord = IP.pibasis.inner[i].orders[j]
    if ord != korr_ord
        Cs[i][j] = 0
    end
  end
  return PIPotential(IP.pibasis, Cs)
end


end  # end of module Pure