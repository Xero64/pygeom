from numpy import isclose, sqrt

from pygeom.geom3d import Vector

x1, y1, z1 = 2.0, 3.0, 1.0

vec1 = Vector(x1, y1, z1)

vec2 = vec1*2

vec3 = Vector(vec2.y, vec2.z, vec2.z)

vec4 = vec1.cross(vec2)

vec5 = vec1.cross(vec3)

vec6 = Vector(vec1.y*vec3.z - vec1.z*vec3.y,
              vec1.z*vec3.x - vec1.x*vec3.z,
              vec1.x*vec3.y - vec1.y*vec3.x)

vec7 = Vector(vec5.x/vec5.return_magnitude(),
              vec5.y/vec5.return_magnitude(),
              vec5.z/vec5.return_magnitude())

def test_vector_multiplication():
    assert Vector.is_close(vec2*5, Vector(10*x1, 10*y1, 10*z1), atol=1e-12)

def test_vector_addition():
    assert Vector.is_close(vec1+vec2, Vector(3*x1, 3*y1, 3*z1), atol=1e-12)

def test_vector_subtraction():
    assert Vector.is_close(vec1-vec2, Vector(-x1, -y1, -z1), atol=1e-12)

def test_vector_dot_product():
    assert isclose(vec1.dot(vec2), 2*x1*x1 + 2*y1*y1 + 2*z1*z1, atol=1e-12)

def test_vector_cross_product_1():
    assert Vector.is_close(vec4, Vector(0.0, 0.0, 0.0), atol=1e-12)

def test_vector_magnitude():
    assert isclose(vec1.return_magnitude(), sqrt(x1**2 + y1**2 + z1**2), atol=1e-12)

def test_vector_cross_product_2():
    assert Vector.is_close(vec5, vec6, atol=1e-12)

def test_vector_to_unit():
    assert Vector.is_close(vec5.to_unit(), vec7, atol=1e-12)

def test_vector_summation():
    assert Vector.is_close(sum([vec1, vec2, vec3], Vector(0.0, 0.0, 0.0)),
                          Vector(3*x1 + 2*y1, 3*y1 + 2*z1, 5*z1), atol=1e-12)

def test_vector_negation():
    assert Vector.is_close(-vec1, Vector(-x1, -y1, -z1), atol=1e-12)

def test_vector_division():
    assert Vector.is_close(vec1/2, Vector(x1/2, y1/2, z1/2), atol=1e-12)

def test_vector_equality():
    assert vec1 == Vector(x1, y1, z1)

def test_vector_inequality():
    assert vec1 != vec2
