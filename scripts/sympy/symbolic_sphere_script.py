#%%
# Import Dependencies
from pygeom.symbol3d import SymbolicVector
from sympy import Symbol, cos, pi, sin

#%%
# Create Symbols
radius = Symbol('radius', real=True, positive=True)
u = Symbol('u', real=True)
v = Symbol('v', real=True)

#%%
# Create Expressions
x = radius*cos(2*pi*u)*sin(pi*v)
y = radius*sin(2*pi**u)*sin(pi*v)
z = radius*cos(pi*v)

ruv = SymbolicVector(x, y, z)

print(f'ruv = \n{ruv}\n')

drdu = ruv.diff(u)

print(f'drdu = \n{drdu}\n')

drdv = ruv.diff(v)

print(f'drdv = \n{drdv}\n')
