"""
`module Pure`

This module contains utility functions to obtain the contribution of basis functions of each body-order. 
"""
module Pure

include("../extimports.jl")
include("../aceimports.jl")
using ACE

export nB_energy

function nB_energy(IP, atoms, Border)
    I_ord = findall(get_orders(IP.components[3]) .== Border);

end


end  # end of module Pure