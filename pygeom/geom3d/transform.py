from math import cos, sin

from .vector import Vector


class Transform():
    """3D Transform Class"""
    dirx: Vector = None
    diry: Vector = None
    dirz: Vector = None
    def __init__(self, vecx: Vector, vecxy: Vector) -> None:
        self.dirx = vecx.to_unit()
        vecz = vecx.cross(vecxy)
        vecy = vecz.cross(vecx)
        self.diry = vecy.to_unit()
        self.dirz = vecz.to_unit()
    def vector_to_global(self, vec: Vector) -> Vector:
        """Transforms a vector from this local coordinate to global."""
        dirx = Vector(self.dirx.x, self.diry.x, self.dirz.x)
        diry = Vector(self.dirx.y, self.diry.y, self.dirz.y)
        dirz = Vector(self.dirx.z, self.diry.z, self.dirz.z)
        x = dirx*vec
        y = diry*vec
        z = dirz*vec
        return Vector(x, y, z)
    def vector_to_local(self, vec: Vector) -> Vector:
        """Transforms a vector from global to this local coordinate."""
        dirx = Vector(self.dirx.x, self.dirx.y, self.dirx.z)
        diry = Vector(self.diry.x, self.diry.y, self.diry.z)
        dirz = Vector(self.dirz.x, self.dirz.y, self.dirz.z)
        x = dirx*vec
        y = diry*vec
        z = dirz*vec
        return Vector(x, y, z)
    def rotate_about_z(self, angle: float) -> 'Transform':
        """Creates a Transform that is rotate about z by an angle [radians]."""
        cos_ang = cos(angle)
        sin_ang = sin(angle)
        dirz = self.dirz
        dirx = self.dirx*cos_ang+self.diry*sin_ang
        diry = self.diry*cos_ang-self.dirx*sin_ang
        return Transform(dirx, diry, dirz)
    def __repr__(self) -> str:
        return '<Transform>'
