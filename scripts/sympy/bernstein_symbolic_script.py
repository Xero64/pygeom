#%%
# Import Dependencies
from sympy import Expr, Symbol

from pygeom.symbol2d import SymbolicVector2D, empty_symbolicvector2d
from pygeom.tools.bernstein import (symbolic_bernstein_first_derivatives,
                                    symbolic_bernstein_polynomials,
                                    symbolic_bernstein_second_derivatives)

#%%
# Create Symbols
x = Symbol('x', real=True)
xa = Symbol('xa', real=True)
xb = Symbol('xb', real=True)
xc = Symbol('xc', real=True)
xd = Symbol('xd', real=True)
ya = Symbol('ya', real=True)
yb = Symbol('yb', real=True)
yc = Symbol('yc', real=True)
yd = Symbol('yd', real=True)
wb = Symbol('wb', real=True)
wc = Symbol('wc', real=True)
t = Symbol('t', real=True)

#%%
# Create Expressions
syms = empty_symbolicvector2d(3, dtype=object)
syms[0] = SymbolicVector2D(xa, ya)
syms[1] = SymbolicVector2D(xb, yb)*wb
syms[2] = SymbolicVector2D(xc, yc)
# syms[3] = SymbolicVector2D(xd, yd)

poly = symbolic_bernstein_polynomials(2, t)
expr: SymbolicVector2D = sum(syms*poly, SymbolicVector2D(0, 0))
expr.__class__ = SymbolicVector2D
print(f'expr = {expr}\n')

dpoly = symbolic_bernstein_first_derivatives(2, t)
dexpr: SymbolicVector2D = sum(syms*dpoly, SymbolicVector2D(0, 0))
dexpr.__class__ = SymbolicVector2D
print(f'dexpr = {dexpr}\n')

d2poly = symbolic_bernstein_second_derivatives(2, t)
d2expr: SymbolicVector2D = sum(syms*d2poly, SymbolicVector2D(0, 0))
d2expr.__class__ = SymbolicVector2D
print(f'd2expr = {d2expr}\n')

demag: Expr = dexpr.return_magnitude()

# print(f'demag = {demag}\n')

demag = demag.expand()

print(f'demag = {demag}\n')

numer = dexpr.cross(d2expr)
denom: Expr = dexpr.return_magnitude()**3

# print(f'numer = {numer}\n')
# print(f'denom = {denom}\n')

numer = numer.expand()
denom = denom.expand()

print(f'numer = {numer}\n')
print(f'denom = {denom}\n')
