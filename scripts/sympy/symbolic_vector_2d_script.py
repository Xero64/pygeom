#%%
# Import Depenedencies
from sympy import Symbol, cos, sin

from pygeom.symbol2d import SymbolicVector2D

#%%
# Create Symbols
th = Symbol('th', real=True)

#%%
# Create Symbolic Vector
vec = SymbolicVector2D(cos(th), sin(th))
print(f'vec = {vec}\n')

mag = vec.return_magnitude()
print(f'mag = {mag}\n')

unit = vec.to_unit()
print(f'unit = {unit}\n')

unit = unit.simplify()
print(f'unit = {unit}\n')
