from numpy import isclose, sqrt

from pygeom.geom2d import Vector2D

x1, y1 = 2.0, 3.0

vec1 = Vector2D(x1, y1)

vec2 = vec1*2

vec3 = Vector2D(vec2.y, vec2.x)

val4 = vec1.cross(vec2)

val5 = vec1.cross(vec3)

val6 = vec1.x*vec3.y - vec1.y*vec3.x

vec7 = Vector2D(vec3.x/vec3.return_magnitude(), vec3.y/vec3.return_magnitude())

def test_vector_multiplication():
    assert Vector2D.is_close(vec2*5, Vector2D(10*x1, 10*y1), atol=1e-12)

def test_vector_addition():
    assert Vector2D.is_close(vec1 + vec2, Vector2D(3*x1, 3*y1), atol=1e-12)

def test_vector_subtraction():
    assert Vector2D.is_close(vec1 - vec2, Vector2D(-x1, -y1), atol=1e-12)

def test_vector_dot_product():
    assert isclose(vec1.dot(vec2), 2*x1*x1 + 2*y1*y1, atol=1e-12)

def test_vector_cross_product_1():
    assert isclose(val4, 0.0, atol=1e-12)

def test_vector_magnitude():
    assert isclose(vec1.return_magnitude(), sqrt(x1**2 + y1**2), atol=1e-12)

def test_vector_cross_product_2():
    assert isclose(val5, val6, atol=1e-12)

def test_vector_to_unit():
    assert Vector2D.is_close(vec3.to_unit(), vec7, atol=1e-12)

def test_vector_summation():
    assert Vector2D.is_close(sum([vec1, vec2, vec3], Vector2D(0.0, 0.0)),
                            Vector2D(3*x1 + 2*y1, 3*y1 + 2*x1), atol=1e-12)

def test_vector_negation():
    assert Vector2D.is_close(-vec1, Vector2D(-x1, -y1), atol=1e-12)

def test_vector_division():
    assert Vector2D.is_close(vec1/2, Vector2D(x1/2, y1/2), atol=1e-12)

def test_vector_equality():
    assert vec1 == Vector2D(x1, y1)

def test_vector_inequality():
    assert vec1 != Vector2D(x1, y1 + 1)
