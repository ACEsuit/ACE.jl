
import sympy as sp
from sympy import Symbol, re, im, I

# x = Symbol('x', real=True)
# y = Symbol('y', real=True)

# a basis function is defined by an l and an m tuple:
ll = [1, 3, 3, 4]
mm = [1, -2, -1, 1]

# the following function takes these tuples and converts them from
# a complex SH basis to a real SH basis
# def c2rsh(ll, mm):

# first we need the required symbols for

C = [ Symbol(f'C{j}', real=True) for j in range(0, len(ll)+1) ]
S = [ Symbol(f'S{j}', real=True) for j in range(0, len(ll)+1) ]
A = [ None for j in range(0, 2*len(ll)+1) ]
A[len(ll)] = C[0]
prodA = C[0]
for j in range(1, len(ll)+1):
    A[len(ll)-j] =  (1/sp.sqrt(2)) * (C[j] - I * S[j])
    A[len(ll)+j] =  ((-1)**j/sp.sqrt(2)) * (C[j] + I * S[j])
    prodA = prodA * A[len(ll)-j] * A[len(ll)+j]

c = Symbol('a', real=True) + I *  Symbol('b', real=True)
print(sp.simplify(sp.re(c * prodA)))
