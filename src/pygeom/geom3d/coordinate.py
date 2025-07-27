from .transform import Transform
from .vector import Vector


class Coordinate(Transform):
    """Coordinate3D Class"""
    pnt: Vector = None

    def __init__(self, pnt: Vector, vecx: Vector, vecxy: Vector) -> None:
        super().__init__(vecx, vecxy)
        self.pnt = pnt

    def point_to_global(self, pnt: Vector) -> Vector:
        """Transforms a point from this local coordinate to global."""
        pnts = self.vector_to_global(pnt) + self.pnt
        return pnt.__class__(pnts.x, pnts.y, pnts.z)

    def point_to_local(self, pnt: Vector) -> Vector:
        """Transforms a point from global to this local coordinate."""
        pnts = self.vector_to_local(pnt - self.pnt)
        return pnt.__class__(pnts.x, pnts.y, pnts.z)

    @classmethod
    def from_3_points_xy(cls: 'Coordinate', pnta: Vector,
                         pntb: Vector, pntc: Vector) -> 'Coordinate':
        pnt = (pnta + pntb + pntc)/3
        vecab = pntb - pnta
        vecbc = pntc - pntb
        return cls(pnt, vecab, vecbc)

    def __repr__(self) -> str:
        return '<Coordinate>'


def coordinate_from_3_points_xy(pnta: Vector, pntb: Vector,
                                pntc: Vector) -> Coordinate:
        pnt = (pnta + pntb + pntc)/3
        vecab = pntb - pnta
        vecbc = pntc - pntb
        return Coordinate(pnt, vecab, vecbc)
