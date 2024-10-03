from math import sqrt

from pygeom.geom3d import Transform
from pygeom.geom3d.vector import Vector, vector_isclose

x = 5.0
y = 8.0
z = 3.0
vec = Vector(x, y, z)

trans = Transform(vec, Vector(0.0, 0.0, 1.0))

vecl = Vector(sqrt(x**2 + y**2 + z**2), 0.0, 0.0)

def test_transform_to_local():
    assert vector_isclose(trans.vector_to_local(vec), vecl)
