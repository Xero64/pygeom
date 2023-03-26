from math import sqrt
from pygeom.geom2d.vector2d import Vector2D, vector2d_isclose
from pygeom.geom2d import Transform2D

x = 5.0
y = 8.0
vec = Vector2D(x, y)

trans = Transform2D(vec)

vecl = Vector2D(sqrt(x**2 + y**2), 0.0)

def test_transform_to_local():
    assert vector2d_isclose(trans.vector_to_local(vec), vecl)
