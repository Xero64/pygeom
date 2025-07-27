from .transform2d import Transform2D
from .vector2d import Vector2D


class Coordinate2D(Transform2D):
    """Coordinate2D Class"""
    pnt: Vector2D = None

    def __init__(self, pnt: Vector2D, dirx: Vector2D) -> None:
        super().__init__(dirx)
        self.pnt = pnt

    def point2d_to_global(self, pnt: Vector2D) -> Vector2D:
        """Transforms a point from this local coordinate system to global."""
        pnts = self.vector2d_to_global(pnt) + self.pnt
        return pnt.__class__(pnts.x, pnts.y)

    def point2d_to_local(self, pnt: Vector2D) -> Vector2D:
        """Transforms a point from global to this local coordinate system."""
        pnts = self.vector2d_to_local(pnt - self.pnt)
        return pnt.__class__(pnts.x, pnts.y)

    def __repr__(self) -> str:
        return '<Coordinate2D>'
