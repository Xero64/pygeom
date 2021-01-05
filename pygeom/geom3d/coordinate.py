from .vector import Vector, vector_from_points
from math import radians, cos, sin

class Coordinate(object):
    """Coordinate Class"""
    pnt = None
    dirx = None
    diry = None
    dirz = None
    def __init__(self, pnt, vx, vy, vz):
        self.pnt = pnt
        self.dirx = vx.to_unit()
        self.diry = vy.to_unit()
        self.dirz = vz.to_unit()
    def vector_to_global(self, vec):
        """Transforms a vector from this local coordinate system to global"""
        dirx = Vector(self.dirx.x, self.diry.x, self.dirz.x)
        diry = Vector(self.dirx.y, self.diry.y, self.dirz.y)
        dirz = Vector(self.dirx.z, self.diry.z, self.dirz.z)
        x = dirx*vec
        y = diry*vec
        z = dirz*vec
        return Vector(x, y, z)
    def vector_to_local(self, vec):
        """Transforms a vector from global to this local coordinate system"""
        dirx = Vector(self.dirx.x, self.dirx.y, self.dirx.z)
        diry = Vector(self.diry.x, self.diry.y, self.diry.z)
        dirz = Vector(self.dirz.x, self.dirz.y, self.dirz.z)
        x = dirx*vec
        y = diry*vec
        z = dirz*vec
        return Vector(x, y, z)
    def point_to_global(self, pnt):
        """Transforms a point from this local coordinate system to global"""
        vecl = pnt.to_vector()
        vecg = self.vector_to_global(vecl)
        pntg = self.pnt+vecg
        return pntg
    def point_to_local(self, pnt):
        """Transforms a point from global  to this local coordinate system"""
        vecg = vector_from_points(self.pnt, pnt)
        vecl = self.vector_to_local(vecg)
        pntl = vecl.to_point()
        return pntl
    def rotate_about_z(self, angle):
        """Creates a coordinate system that is rotate about x by an angle [deg]"""
        angle = radians(angle)
        cos_ang = cos(angle)
        sin_ang = sin(angle)
        pnt = self.pnt
        dirz = self.dirz
        tx = self.dirx*cos_ang+self.diry*sin_ang
        ty = self.diry*cos_ang-self.dirx*sin_ang
        return Coordinate(pnt, tx, ty, dirz)
    def __repr__(self):
        return '<Coordinate>'

def coordinate_from_points(pnta, pntb, pntc):
    """Create a Coordinate from three Points"""
    pnt = pnta
    dirx = vector_from_points(pnta, pntb).to_unit()
    dirxy = vector_from_points(pnta, pntc).to_unit()
    dirz = dirx**dirxy
    diry = dirz**dirx
    return Coordinate(pnt, dirx, diry, dirz)
