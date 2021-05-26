"""
`module Pure`

This module contains utility functions to obtain the contribution of basis functions of each body-order. 
"""
module Pure

include("../extimports.jl")
include("../aceimports.jl")
#using ACE
using JuLIP

export nB_energy

function nB_energy(IP::PIPotential, atoms::Atoms, B_order::Int)
  """
  This function returns the Body ordered contributions to the total ACE energy
    Arguments:
      IP: PIPotential object
      atoms: Atoms object whose energy should be returned
      B_order: Int, specifying the body order (correlation order + 1)
  """
  korr_ord = B_order - 1
  # get the correlation orders of the PIbasis ...
  Ns_pi = zeros(Int, length(IP.pibasis))
  for iz0 = 1:numz(IP)
     inner = IP.pibasis.inner[iz0]
     Ns_pi[inner.AAindices] .= inner.orders
  end
  Cs = deepcopy(IP.coeffs)
  coeff_size = [length(c) for c in Cs]
  ind = 1
  for i in 1:length(coeff_size)
    for j in 1:coeff_size[i]
        if Ns_pi[ind] != korr_ord
            Cs[i][j] = 0
        end
        ind += 1
    end
  end
  IP_new = PIPotential(IP.pibasis, Cs)
  return energy(IP_new, atoms)

end


end  # end of module Pure