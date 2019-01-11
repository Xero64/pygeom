from .point2d import Point2D
from .vector2d import Vector2D

class InfLine2D(object):
    pnt = None
    uvec = None
    def __init__(self, pnt, uvec):
        self.pnt = pnt
        self.uvec = uvec.to_unit()
    def point_along_vector(self, length):
        x = self.pnt.x+self.uvec.x*length
        y = self.pnt.y+self.uvec.y*length
        return Point2D(x, y)
    def __repr__(self):
        return '<InfLine2D>'

def intersection_of_inflines2D(iln1, iln2, tol=1e-12):
    uv1 = iln1.uvec
    uv2 = iln2.uvec
    xp12 = uv1**uv2
    if abs(xp12) < tol:
        return None
    pt1 = iln1.pnt
    pt2 = iln2.pnt
    x21 = pt2.x-pt1.x
    y21 = pt2.y-pt1.y
    l1 = (uv2.y*x21-uv2.x*y21)/xp12
    xp = pt1.x + l1*uv1.x
    yp = pt1.y + l1*uv1.y
    return Point2D(xp, yp)
