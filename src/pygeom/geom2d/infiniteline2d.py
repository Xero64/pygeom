from .vector2d import Vector2D


class InfiniteLine2D():
    pnt: Vector2D = None
    uvec: Vector2D = None

    def __init__(self, pnt: Vector2D, uvec: Vector2D) -> None:
        self.pnt = pnt
        self.uvec = uvec.to_unit()

    def point_along_vector(self, length: float) -> Vector2D:
        x = self.pnt.x + self.uvec.x*length
        y = self.pnt.y + self.uvec.y*length
        return Vector2D(x, y)

    def __repr__(self) -> str:
        return '<InfiniteLine2D>'

def intersection_length_of_ifl2D(iln1: InfiniteLine2D, iln2: InfiniteLine2D,
                                 tol: float=1e-12) -> float:
    uv1 = iln1.uvec
    uv2 = iln2.uvec
    xp12 = uv1.cross(uv2)
    if abs(xp12) < tol:
        raise ValueError('The lines are parallel.')
    pt1 = iln1.pnt
    pt2 = iln2.pnt
    x21 = pt2.x - pt1.x
    y21 = pt2.y - pt1.y
    l1 = (uv2.y*x21 - uv2.x*y21)/xp12
    return l1

def intersection_of_ifl2D(iln1: 'InfiniteLine2D', iln2: 'InfiniteLine2D',
                          tol: float=1e-12) -> Vector2D:
    l1 = intersection_length_of_ifl2D(iln1, iln2, tol=tol)
    xp = iln1.pnt.x + l1*iln1.uvec.x
    yp = iln1.pnt.y + l1*iln1.uvec.y
    return Vector2D(xp, yp)
