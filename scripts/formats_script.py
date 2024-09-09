#%%
# Import Dependencies
from pygeom.geom3d import Point, Vector

#%%
# Create Point and Vector
pnt = Point(1.5634356746, 45.2354356425, 1.4352542534)
vec = Vector(1.5634356746, 45.2354356425, 1.4352542534)

#%%
# Print Point and Vector
print(f'pnt = {pnt}')
print(f'pnt = {pnt:g}')
print(f'pnt = {pnt:.2f}')
print(f'pnt = {pnt:}')
print(f'pnt = {pnt:.5e}')

print(f'vec = {vec}')
print(f'vec = {vec:g}')
print(f'vec = {vec:.2f}')
print(f'vec = {vec:}')
print(f'vec = {vec:.5e}')
