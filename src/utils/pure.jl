"""
`module Pure`

This module contains utility functions to obtain the contribution of basis functions of each body-order. 
"""
module Pure

include("../extimports.jl")
include("../aceimports.jl")
#using ACE

export nB_energy

function nB_energy(IP, atoms, B_order)
  korr_ord = B_order - 1
  #I_ord = findall(get_orders(IP.components[3]) .== Border);
  # get the correlation orders of the PIbasis ...
  Ns_pi = zeros(Int, length(IP.components[3].pibasis))
  for iz0 = 1:numz(IP.components[3])
     inner = IP.components[3].pibasis.inner[iz0]
     Ns_pi[inner.AAindices] .= inner.orders
  end
  IP.components[3].coeffs
  coeff_size = [length(c) for c in coeffs]
  ind = 1
  for i in 1:length(coeff_size)
    for j in 1:coeff_size[i]
        if Ns_pi[ind] != korr_ord
            IP.components[3].coeffs[i][j] = 0
        end
        ind += 1
    end
  end
  return nB_energy(IP.components[3], atoms)

end


end  # end of module Pure