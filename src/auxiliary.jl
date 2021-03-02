
# --------------------------------------------------------------------------
# ACE.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# ------------------------------------------------------------
# miscellaneous

"""
a simple utility function to check whether two objects are equal
"""
_allfieldsequal(x1, x2) =
      all( getfield(x1, sym) == getfield(x2, sym)
           for sym in union(fieldnames(typeof(x1)), fieldnames(typeof(x2))) )
