#%%
# Import Dependencies
from sympy import Symbol, cos, pi, sin

from pygeom.symbol3d import SymbolicVector

#%%
# Create Symbols
ra = Symbol('ra', real=True, positive=True)
rm = Symbol('rm', real=True, positive=True)
u = Symbol('u', real=True)
v = Symbol('v', real=True)

#%%
# Create Expressions
x = cos(2*pi*u)*(ra*cos(2*pi*v) + rm)
y = sin(2*pi*u)*(ra*cos(2*pi*v) + rm)
z = ra*sin(2*pi*v)

ruv = SymbolicVector(x, y, z)

print(f'ruv = \n{ruv}\n')

drdu = ruv.diff(u)

print(f'drdu = \n{drdu}\n')

drdv = ruv.diff(v)

print(f'drdv = \n{drdv}\n')
