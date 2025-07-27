from .vector import Vector


class InfiniteLine3D():
    pnt: Vector = None
    uvec: Vector = None

    def __init__(self, pnt: Vector, uvec: Vector) -> None:
        self.pnt = pnt
        self.uvec = uvec.to_unit()

    def point_along_vector(self, length: float) -> Vector:
        x = self.pnt.x + self.uvec.x*length
        y = self.pnt.y + self.uvec.y*length
        z = self.pnt.z + self.uvec.z*length
        return Vector(x, y, z)

    def __repr__(self) -> str:
        return '<InfiniteLine3D>'

def min_dist_between_ifl3D(iln1: InfiniteLine3D, iln2: InfiniteLine3D,
                           tol: float=1e-12) -> tuple[float, float, float]:
    ux12 = iln2.uvec.cross(iln1.uvec)
    ux12m = ux12.return_magnitude()
    if ux12m < tol:
        raise ValueError('The lines are parallel.')
    p21 = iln2.pnt - iln1.pnt
    l1 = iln2.uvec.cross(p21).dot(ux12)
    l2 = iln1.uvec.cross(p21).dot(ux12)
    d = p21.dot(ux12)
    return d, l1, l2

def pnts_min_dist_between_ifl3D(iln1: InfiniteLine3D, iln2: InfiniteLine3D,
                                tol: float=1e-12) -> tuple[Vector,Vector]:
    _, l1, l2 = min_dist_between_ifl3D(iln1, iln2, tol=tol)
    p1 = iln1.pnt + l1*iln1.uvec
    p2 = iln2.pnt + l2*iln2.uvec
    return p1, p2
