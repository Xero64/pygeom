from .point import Point
from .vector import Vector, vector_from_points

class Plane(object):
    """Plane Class"""
    pnt = None
    nrm = None
    def __init__(self, pnt, nrm):
        self.pnt = pnt
        self.nrm = nrm.to_unit()
    def return_abcd(self):
        a = self.nrm.x
        b = self.nrm.y
        c = self.nrm.z
        d = -a*self.pnt.x-b*self.pnt.y-c*self.pnt.z
        return a, b, c, d
    def point_z_from_plane(self, pnt):
        vec = vector_from_points(self.pnt, pnt)
        return vec*self.nrm
    def reverse_normal(self):
        self.nrm = -self.nrm
    def __repr__(self):
        return '<Plane>'

def plane_from_multiple_points(pnts):
    """Create a Plane from three Points"""
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
    D = sxx*syy-sxy**2
    a = (syz*sxy-sxz*syy)/D
    b = (sxy*sxz-sxx*syz)/D
    nrm = Vector(a, b, 1.)
    return Plane(pntc, nrm)
