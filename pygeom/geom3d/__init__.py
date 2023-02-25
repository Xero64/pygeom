from typing import Iterable, List

from .vector import Vector, vector_from_points
from .point import Point
from .coordinate import Coordinate
from .plane import Plane

IHAT = Vector(1.0, 0.0, 0.0)
JHAT = Vector(0.0, 1.0, 0.0)
KHAT = Vector(0.0, 0.0, 1.0)

def coordinate_from_points(pnta: 'Vector', pntb: 'Vector',
                           pntc: 'Vector') -> 'Coordinate':
    """Create a Coordinate from three Points"""
    pnt = pnta
    dirx = vector_from_points(pnta, pntb).to_unit()
    dirxy = vector_from_points(pnta, pntc).to_unit()
    dirz = dirx**dirxy
    diry = dirz**dirx
    return Coordinate(pnt, dirx, diry, dirz)

def plane_from_multiple_points(pnts: Iterable['Point']) -> 'Plane':
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

def point_from_lists(x: Iterable['float'], y: Iterable['float'],
                     z: Iterable['float']) -> List['Point']:
    """Create a list of Point objects"""
    return [Point(x[i], y[i], z[i]) for i in range(len(x))]

def midpoint_of_points(pnts: Iterable['Point']) -> 'Point':
    num = len(pnts)
    x = sum(pnt.x for pnt in pnts)/num
    y = sum(pnt.y for pnt in pnts)/num
    z = sum(pnt.z for pnt in pnts)/num
    return Point(x, y, z)