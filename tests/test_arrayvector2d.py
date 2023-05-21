from numpy import array, isclose, sqrt
from pygeom.array2d import ArrayVector2D, arrayvector2d_allclose

x1, y1 = array(2.0), array(3.0)

vec1 = ArrayVector2D(x1, y1)

vec2 = vec1*2

vec3 = ArrayVector2D(vec2.y, vec2.x)

val4 = vec1.cross(vec2)

val5 = vec1.cross(vec3)

val6 = vec1.x*vec3.y - vec1.y*vec3.x

vec7 = ArrayVector2D(vec3.x/vec3.return_magnitude(), vec3.y/vec3.return_magnitude())

print(arrayvector2d_allclose(vec2*5, ArrayVector2D(10*x1, 10*y1), atol=1e-12))

def test_vector_multiplication():
    assert arrayvector2d_allclose(vec2*5, ArrayVector2D(10*x1, 10*y1), atol=1e-12)

def test_vector_addition():
    assert arrayvector2d_allclose(vec1 + vec2, ArrayVector2D(3*x1, 3*y1), atol=1e-12)

def test_vector_subtraction():
    assert arrayvector2d_allclose(vec1 - vec2, ArrayVector2D(-x1, -y1), atol=1e-12)

def test_vector_dot_product():
    assert isclose(vec1.dot(vec2), 2*x1*x1 + 2*y1*y1, atol=1e-12)

def test_vector_cross_product_1():
    assert isclose(val4, 0.0, atol=1e-12)

def test_vector_magnitude():
    assert isclose(vec1.return_magnitude(), sqrt(x1**2 + y1**2), atol=1e-12)

def test_vector_cross_product_2():
    assert isclose(val5, val6, atol=1e-12)

def test_vector_unit_vector():
    assert arrayvector2d_allclose(vec3.to_unit(), vec7, atol=1e-12)

def test_vector_negation():
    assert arrayvector2d_allclose(-vec1, ArrayVector2D(-x1, -y1), atol=1e-12)

def test_vector_division():
    assert arrayvector2d_allclose(vec1/2, ArrayVector2D(x1/2, y1/2), atol=1e-12)

def test_vector_equality():
    assert vec1 == ArrayVector2D(x1, y1)

def test_vector_inequality():
    assert vec1 != ArrayVector2D(x1, y1 + 1)
