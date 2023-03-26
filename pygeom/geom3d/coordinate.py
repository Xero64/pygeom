from math import cos, sin

from . import vector_from_points
from .point import Point
from .transform import Transform
from .vector import Vector


class Coordinate(Transform):
    """3D Coordinate Class"""
    pnt: Point = None
    def __init__(self, pnt: Point, vecx: Vector,
                 vecxy: Vector) -> None:
        super().__init__(vecx, vecxy)
        self.pnt = pnt
    def point_to_global(self, pnt: Point) -> Point:
        """Transforms a point from this local coordinate to global."""
        vecl = pnt.to_vector()
        vecg = self.vector_to_global(vecl)
        pntg = self.pnt+vecg
        return pntg
    def point_to_local(self, pnt: Point) -> Point:
        """Transforms a point from global  to this local coordinate."""
        vecg = vector_from_points(self.pnt, pnt)
        vecl = self.vector_to_local(vecg)
        pntl = vecl.to_point()
        return pntl
    def rotate_about_z(self, angle: float) -> 'Coordinate':
        """Creates a coordinate that is rotate about z by an angle [radians]."""
        cos_ang = cos(angle)
        sin_ang = sin(angle)
        pnt = self.pnt
        dirx = self.dirx*cos_ang + self.diry*sin_ang
        diry = self.diry*cos_ang - self.dirx*sin_ang
        return Coordinate(pnt, dirx, diry)
    def __repr__(self) -> str:
        return '<Coordinate>'
