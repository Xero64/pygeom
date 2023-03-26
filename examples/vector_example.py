#%%
# Import Vector Class
from pygeom.geom3d import Vector

# Vector object creation
vec1 = Vector(2.0, 3.0, 1.0)
print(f'vec1 = {vec1:g}')

# Multiply vector by a scalar
vec2 = 2*vec1
print(f'vec2 = {vec2:g}')

# Vector attributes are x, y, and z
vec3 = Vector(vec2.y, vec2.z, vec2.z)
print(f'vec3 = {vec3:.6f}')

# Vector cross product
vec4 = vec1**vec2
print(f'vec4 = {vec4:g}')

# Vector cross product
vec5 = vec1**vec3
print(f'vec5 = {vec5:g}')

# Vector dot product
scal = vec1*vec2
print(f'scal = {scal:g}')

# Get vector magnitude
mag = vec5.return_magnitude()
print(f'mag = {mag:g}')

# Get the unit vector
uvec = vec5.to_unit()
print(f'uvec = {uvec:.5f}')
