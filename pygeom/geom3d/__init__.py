from typing import Iterable, List

from math import atan2
from .vector import Vector
from .point import Point
from .transform import Transform as Transform
from .coordinate import Coordinate
from .cubicspline import CubicSpline as CubicSpline
from .line import Line as Line
from .infiniteline3d import InfiniteLine3D as InfiniteLine3D
from .plane import Plane

IHAT = Vector(1.0, 0.0, 0.0)
JHAT = Vector(0.0, 1.0, 0.0)
KHAT = Vector(0.0, 0.0, 1.0)

def coordinate_from_points(pnta: Vector, pntb: Vector,
                           pntc: Vector) -> Coordinate:
    """Create a Coordinate from three Points"""
    pnt = pnta
    vecx = pntb - pnta
    vecxy = pntc - pnta
    return Coordinate(pnt, vecx, vecxy)

def plane_from_multiple_points(pnts: Iterable[Point]) -> Plane:
    """Create a Plane from multiple Points"""
    n = len(pnts)
    x = [pnt.x for pnt in pnts]
    y = [pnt.y for pnt in pnts]
    z = [pnt.z for pnt in pnts]
    pntc = Point(sum(x)/n, sum(y)/n, sum(z)/n)
    x = [pnt.x-pntc.x for pnt in pnts]
    y = [pnt.y-pntc.y for pnt in pnts]
    z = [pnt.z-pntc.z for pnt in pnts]
    sxx = sum([x[i]**2 for i in range(n)])
    sxy = sum([x[i]*y[i] for i in range(n)])
    sxz = sum([x[i]*z[i] for i in range(n)])
    syy = sum([y[i]**2  for i in range(n)])
    syz = sum([y[i]*z[i] for i in range(n)])
    d = sxx*syy-sxy**2
    a = (syz*sxy-sxz*syy)/d
    b = (sxy*sxz-sxx*syz)/d
    nrm = Vector(a, b, 1.0)
    return Plane(pntc, nrm)

def point_from_lists(x: Iterable[float], y: Iterable[float],
                     z: Iterable[float]) -> List['Point']:
    """Create a list of Point objects"""
    return [Point(x[i], y[i], z[i]) for i in range(len(x))]

def midpoint_of_points(pnts: Iterable[Point]) -> Point:
    """Calculate the midpoint of a list of Points"""
    num = len(pnts)
    x = sum(pnt.x for pnt in pnts)/num
    y = sum(pnt.y for pnt in pnts)/num
    z = sum(pnt.z for pnt in pnts)/num
    return Point(x, y, z)

def solid_angle_tetrahedron(va: Vector, vb: Vector, vc: Vector) -> float:
    """Calculate the solid angle of a tetrahedron pyramid.

    Args:
        va (Vector): First vertex of the tetrahedron pyramid.
        vb (Vector): Second vertex of the tetrahedron pyramid.
        vc (Vector): Third vertex of the tetrahedron pyramid.

    Returns:
        float: Solid angle of the triangular pyramid.
    """
    a = va.return_magnitude()
    b = vb.return_magnitude()
    c = vc.return_magnitude()

    num = va.dot(vb.cross(vc))
    den = a*b*c + a*vb.dot(vc) + b*va.dot(vc) + c*va.dot(vb)

    return 2*atan2(num, den)

#%%
# Solid Angle of Trapezoidal Pyramid
def solid_angle_apex_trapzpyr(va: Vector, vb: Vector, vc: Vector, vd: Vector) -> float:
    """Calculate the solid angle of the apex of a trapezoidal pyramid.

    Args:
        va (Vector): First vertex of the trapezoidal pyramid.
        vb (Vector): Second vertex of the trapezoidal pyramid.
        vc (Vector): Third vertex of the trapezoidal pyramid.
        vd (Vector): Fourth vertex of the trapezoidal pyramid.

    Returns:
        float: Solid angle of apex of the trapezoidal pyramid.
    """
    ve = va + vb + vc + vd

    sa_ab = solid_angle_tetrahedron(va, vb, ve)
    sa_bc = solid_angle_tetrahedron(vb, vc, ve)
    sa_cd = solid_angle_tetrahedron(vc, vd, ve)
    sa_da = solid_angle_tetrahedron(vd, va, ve)

    return sa_ab + sa_bc + sa_cd + sa_da
