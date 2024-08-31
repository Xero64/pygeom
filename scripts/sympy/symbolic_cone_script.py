#%%
# Import Dependencies
from pygeom.symbol3d import SymbolicVector
from sympy import Symbol, cos, pi, sin

#%%
# Create Symbols
radius = Symbol('radius', real=True, positive=True)
height = Symbol('height', real=True, positive=True)
u = Symbol('u', real=True)
v = Symbol('v', real=True)

#%%
# Create Expressions
x = radius*(1 - v)*cos(2*pi*u)
y = radius*(1 - v)*sin(2*pi*u)
z = height*v

ruv = SymbolicVector(x, y, z)

print(f'ruv = \n{ruv}\n')

drdu = ruv.diff(u)

print(f'drdu = \n{drdu}\n')

drdv = ruv.diff(v)

print(f'drdv = \n{drdv}\n')
