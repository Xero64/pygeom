from .vector2d import Vector2D, vector2d_from_points

class Coordinate2D(object):
    """Coordinate2D Class"""
    pnt = None
    dirx = None
    diry = None
    def __init__(self, pnt, dirx, diry):
        self.pnt = pnt
        self.dirx = dirx.to_unit()
        self.diry = diry.to_unit()
    def vector_to_global(self, vec):
        """Transforms a vector from this local coordinate system to global"""
        dirx = Vector2D(self.dirx.x, self.diry.x)
        diry = Vector2D(self.dirx.y, self.diry.y)
        x = dirx*vec
        y = diry*vec
        return Vector2D(x, y)
    def vector_to_local(self, vec):
        """Transforms a vector from global  to this local coordinate system"""
        dirx = self.dirx
        diry = self.diry
        x = dirx*vec
        y = diry*vec
        return Vector2D(x, y)
    def point_to_global(self, pnt):
        """Transforms a point from this local coordinate system to global"""
        vecl = pnt.toVector()
        vecg = self.vector_to_global(vecl)
        pntg = self.pnt+vecg
        return pntg
    def point_to_local(self, pnt):
        """Transforms a point from global  to this local coordinate system"""
        vecg = vector2d_from_points(self.pnt, pnt)
        vecl = self.vector_to_local(vecg)
        pntl = vecl.to_point()
        return pntl
    def __repr__(self):
        return '<Coordinate2D>'

def coordinate2d_from_points(pnta, pntb):
    """Create a Vector2D from two Point2Ds"""
    pnt = pnta
    dirx = vector2d_from_points(pnta, pntb)
    diry = Vector2D(-dirx.y, dirx.x)
    return Coordinate2D(pnt, dirx, diry)
