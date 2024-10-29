from math import sqrt

from pygeom.geom2d import Transform2D, Vector2D

x = 5.0
y = 8.0
vec = Vector2D(x, y)

trans = Transform2D(vec)

vecl = Vector2D(sqrt(x**2 + y**2), 0.0)

def test_transform_to_local():
    assert Vector2D.is_close(trans.vector2d_to_local(vec), vecl)
