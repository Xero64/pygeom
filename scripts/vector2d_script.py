#%%
# Import Dependencies
from sympy import Symbol

from pygeom.geom2d.vector2d import Vector2D

#%%
# Create Symbols
x = Symbol('x', real=True)
y = Symbol('y', real=True)

#%%
# Multiply
vec = Vector2D(2.1, 5.0)
print(vec)

newvec = x*vec
print(newvec)

#%%
# Dot Product
vec2 = Vector2D(8*y**2, 5*x*y)
print(vec2)

res = vec2.dot(newvec)
print(res)

#%%
# Cross Product
res = vec2.cross(newvec)
print(res)
