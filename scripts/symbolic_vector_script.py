#%%
# Import Depenedencies
from pygeom.symbol3d import SymbolicVector
from sympy import Symbol, cos, sin

#%%
# Create Symbols
th = Symbol('th', real=True)
al = Symbol('al', real=True)

#%%
# Create Symbolic Vector
vec = SymbolicVector(cos(th)*cos(al), sin(th), cos(th)*sin(al))
print(f'vec = {vec}\n')

mag = vec.return_magnitude()
print(f'mag = {mag}\n')

unit = vec.to_unit()
print(f'unit = {unit}\n')

unit = unit.simplify()
print(f'unit = {unit}\n')
