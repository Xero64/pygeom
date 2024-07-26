# pygeom

Geometry Definition and Calculation Package for Python

Contains:

1. 2D and 3D vectors for vector dot products using "*" and vector cross products using "**".
2. 2D and 3D cubic splines for calculating the various directions and curvatures at every point.
3. 2D and 3D transformations and coordinate systems for transforming vectors and points.

Vector Example Code:

#
``` python
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
print(f'vec3 = {vec3:f}')

# Vector cross product
vec4 = vec1.cross(vec2)
print(f'vec4 = {vec4:g}')

# Vector cross product
vec5 = vec1.cross(vec3)
print(f'vec5 = {vec5:g}')

# Vector dot product
scal = vec1.dot(vec2)
print(f'scal = {scal:g}')

# Get vector magnitude
mag = vec5.return_magnitude()
print(f'mag = {mag:g}')

# Get the unit vector
uvec = vec5.to_unit()
print(f'uvec = {uvec:f}')
```

Vector Example Output:
```
vec1 = <2, 3, 1>
vec2 = <4, 6, 2>
vec3 = <6.000000, 2.000000, 2.000000>
vec4 = <0, 0, 0>
vec5 = <4, 2, -14>
scal = 28
mag = 14.6969
uvec = <0.27217, 0.13608, -0.95258>
```
