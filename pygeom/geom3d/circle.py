from ..geom2d.circle2d import circle2d_from_3_points
from .coordinate import coordinate_from_3_points_xy
from .vector import Vector


class Circle():
    point: Vector = None
    radius: float = None
    normal: Vector = None

    def __init__(self, point: Vector, radius: float, normal: Vector) -> None:
        self.point = point
        self.radius = radius
        self.normal = normal

    def __repr__(self) -> str:
        return f'<Circle: {self.point}, {self.radius}, {self.normal}>'


def circle_from_3_points(pnta: Vector, pntb: Vector,
                         pntc: Vector) -> Circle:
    crd = coordinate_from_3_points_xy(pnta, pntb, pntc)
    lpnta = crd.point_to_local(pnta)
    lpntb = crd.point_to_local(pntb)
    lpntc = crd.point_to_local(pntc)
    circle2d = circle2d_from_3_points(lpnta, lpntb, lpntc)
    lpnt = Vector(circle2d.point.x, circle2d.point.y, 0.0)
    pnt = crd.point_to_global(lpnt)
    return Circle(pnt, circle2d.radius, crd.dirz)
