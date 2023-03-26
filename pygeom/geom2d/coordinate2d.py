from . import vector2d_from_points
from .point2d import Point2D
from .transform2d import Transform2D
from .vector2d import Vector2D


class Coordinate2D(Transform2D):
    """Coordinate2D Class"""
    pnt: 'Point2D' = None
    dirx: 'Vector2D' = None
    def __init__(self, pnt: 'Point2D', dirx: 'Vector2D') -> None:
        super().__init__(dirx)
        self.pnt = pnt
    def point_to_global(self, pnt: 'Point2D') -> 'Vector2D':
        """Transforms a point from this local coordinate system to global."""
        vecg = self.vector_to_global(pnt)
        pntg = self.pnt + vecg
        return pntg
    def point_to_local(self, pnt: 'Point2D') -> 'Vector2D':
        """Transforms a point from global to this local coordinate system."""
        vecg = vector2d_from_points(self.pnt, pnt)
        vecl = self.vector_to_local(vecg)
        pntl = vecl.to_point()
        return pntl
    def __repr__(self) -> str:
        return '<Coordinate2D>'
