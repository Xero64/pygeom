# pygeom

Geometry Definition and Calculation Package for Python

Contains:

1. 2D and 3D vectors for vector dot products using "*" and vector cross products using "**".
2. 2D and 3D cubic splines for calculating the various directions, curvatures, etc. at every point.
3. 2D and 3D coordinate systems for transforming vectors and points.

Vector Example Code:

```python
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
```
Vector Example Output:
```
vec1 = <2, 3.0, 1>
vec2 = <4, 6.0, 2>
vec3 = <6.0, 2, 2>
vec4 = <0.0, 0, 0.0>
vec5 = <4.0, 2.0, -14.0>
scal = 28.0
mag = 14.696938456699069
uvec = <0.2721655269759087, 0.13608276348795434, -0.9525793444156804>
```