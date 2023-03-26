from math import sqrt, isclose
from pygeom.geom2d.vector2d import Vector2D, vector2d_isclose

x1, y1 = 2.0, 3.0

vec1 = Vector2D(x1, y1)

vec2 = vec1*2

vec3 = Vector2D(vec2.y, vec2.x)

val4 = vec1**vec2

val5 = vec1**vec3

val6 = vec1.x*vec3.y - vec1.y*vec3.x

vec7 = Vector2D(vec3.x/vec3.return_magnitude(), vec3.y/vec3.return_magnitude())

def test_vector_multiplication():
    assert vector2d_isclose(vec2*5, Vector2D(10*x1, 10*y1), abs_tol=1e-12)

def test_vector_addition():
    assert vector2d_isclose(vec1+vec2, Vector2D(3*x1, 3*y1), abs_tol=1e-12)

def test_vector_subtraction():
    assert vector2d_isclose(vec1-vec2, Vector2D(-x1, -y1), abs_tol=1e-12)

def test_vector_dot_product():
    assert isclose(vec1*vec2, 2*x1*x1 + 2*y1*y1, abs_tol=1e-12)

def test_vector_cross_product_1():
    assert isclose(val4, 0.0, abs_tol=1e-12)

def test_vector_magnitude():
    assert isclose(vec1.return_magnitude(), sqrt(x1**2 + y1**2), abs_tol=1e-12)

def test_vector_cross_product_2():
    assert isclose(val5, val6, abs_tol=1e-12)

def test_vector_unit_vector():
    assert vector2d_isclose(vec3.to_unit(), vec7, abs_tol=1e-12)

def test_vector_summation():
    assert vector2d_isclose(sum([vec1, vec2, vec3]), Vector2D(3*x1+2*y1, 3*y1+2*x1),
                          abs_tol=1e-12)
def test_vector_negation():
    assert vector2d_isclose(-vec1, Vector2D(-x1, -y1), abs_tol=1e-12)

def test_vector_division():
    assert vector2d_isclose(vec1/2, Vector2D(x1/2, y1/2), abs_tol=1e-12)

def test_vector_equality():
    assert vec1 == Vector2D(x1, y1)

def test_vector_inequality():
    assert vec1 != Vector2D(x1, y1+1)
