#%%

from pygeom import Vector

# Vector object creation
vec1 = Vector(2, 3., 1)
print('vec1 = {:}'.format(vec1))

# Multiply vector by a scalar
vec2 = 2*vec1
print('vec2 = {:}'.format(vec2))

# Vector attributes are x, y, and z
vec3 = Vector(vec2.y, vec2.z, vec2.z)
print('vec3 = {:}'.format(vec3))

# Vector cross product
vec4 = vec1**vec2
print('vec4 = {:}'.format(vec4))

# Vector cross product
vec5 = vec1**vec3
print('vec5 = {:}'.format(vec5))

# Vector dot product
scal = vec1*vec2
print('scal = {:}'.format(scal))

# Get vector magnitude
mag = vec5.return_magnitude()
print('mag = {:}'.format(mag))

# Get the unit vector
uvec = vec5.to_unit()
print('uvec = {:}'.format(uvec))
