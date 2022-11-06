from .vector2d import Vector2D, vector2d_from_points
from .tensor2d import Tensor2D
from .point2d import Point2D

class Coordinate2D():
    """Coordinate2D Class"""
    pnt = None
    dirx = None
    diry = None
    def __init__(self, pnt: 'Point2D', dirx: 'Vector2D') -> None:
        self.pnt = pnt
        self.dirx = dirx.to_unit()
        self.diry = Vector2D(-self.dirx.y, self.dirx.x)
    def vector_to_global(self, vec: 'Vector2D') -> 'Vector2D':
        """Transforms a vector from this local coordinate system to global."""
        dirx = Vector2D(self.dirx.x, self.diry.x)
        diry = Vector2D(self.dirx.y, self.diry.y)
        x = dirx*vec
        y = diry*vec
        return Vector2D(x, y)
    def vector_to_local(self, vec: 'Vector2D') -> 'Vector2D':
        """Transforms a vector from global to this local coordinate system."""
        dirx = self.dirx
        diry = self.diry
        x = dirx*vec
        y = diry*vec
        return Vector2D(x, y)
    def point_to_global(self, pnt: 'Point2D') -> 'Vector2D':
        """Transforms a point from this local coordinate system to global."""
        vecl = pnt.to_vector()
        vecg = self.vector_to_global(vecl)
        pntg = self.pnt + vecg
        return pntg
    def point_to_local(self, pnt: 'Point2D') -> 'Vector2D':
        """Transforms a point from global to this local coordinate system."""
        vecg = vector2d_from_points(self.pnt, pnt)
        vecl = self.vector_to_local(vecg)
        pntl = vecl.to_point()
        return pntl
    def tensor_to_global(self, tens: 'Tensor2D') -> 'Tensor2D':
        """Transforms a vector from this local coordinate system to global."""
        sxx, sxy, syx, syy = tens.to_xy()
        qxx, qxy, qyx, qyy = self.dirx.x, self.dirx.y, self.diry.x, self.diry.y
        exx = qxx**2*sxx + qxx*qyx*sxy + qxx*qyx*syx + qyx**2*syy
        eyy = qxy**2*sxx + qxy*qyy*sxy + qxy*qyy*syx + qyy**2*syy
        exy = qxx*qxy*sxx + qxx*qyy*sxy + qxy*qyx*syx + qyx*qyy*syy
        eyx = qxx*qxy*sxx + qxx*qyy*syx + qxy*qyx*sxy + qyx*qyy*syy
        return Tensor2D(exx, exy, eyx, eyy)
    def tensor_to_local(self, tens: 'Tensor2D') -> 'Tensor2D':
        """Transforms a vector from global to this local coordinate system."""
        sxx, sxy, syx, syy = tens.to_xy()
        qxx, qxy, qyx, qyy = self.dirx.x, self.dirx.y, self.diry.x, self.diry.y
        exx = qxx**2*sxx + qxx*qxy*sxy + qxx*qxy*syx + qxy**2*syy
        eyy = qyx**2*sxx + qyx*qyy*sxy + qyx*qyy*syx + qyy**2*syy
        exy = qxx*qyx*sxx + qxx*qyy*sxy + qxy*qyx*syx + qxy*qyy*syy
        eyx = qxx*qyx*sxx + qxx*qyy*syx + qxy*qyx*sxy + qxy*qyy*syy
        return Tensor2D(exx, exy, eyx, eyy)
    def __repr__(self):
        return '<Coordinate2D>'

def coordinate2d_from_points(pnta: Point2D, pntb: Point2D):
    """Create a Coordinate2D from two Point2Ds."""
    pnt = pnta
    dirx = vector2d_from_points(pnta, pntb)
    diry = Vector2D(-dirx.y, dirx.x)
    return Coordinate2D(pnt, dirx, diry)

def coordinate2d_from_angle(pnt: Point2D, angle: float):
    """Create a Coordinate2D from a Point2D and an Angle."""
    from math import cos, sin
    dirx = Vector2D(cos(angle), sin(angle))
    diry = Vector2D(-dirx.y, dirx.x)
    return Coordinate2D(pnt, dirx, diry)
