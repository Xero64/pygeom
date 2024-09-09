from numpy import sqrt

from .vector2d import Vector2D


class Circle2D():
    point: Vector2D = None
    radius: float = None

    def __init__(self, point: Vector2D, radius: float) -> None:
        self.point = point
        self.radius = radius

    def __repr__(self) -> str:
        return f'<Circle2D: {self.point}, {self.radius}>'


def circle2d_from_3_points(pnta: Vector2D, pntb: Vector2D,
                           pntc: Vector2D) -> Circle2D:
    xa, ya = pnta.x, pnta.y
    xb, yb = pntb.x, pntb.y
    xc, yc = pntc.x, pntc.y
    jac = xa*yb - xa*yc - xb*ya + xb*yc + xc*ya - xc*yb
    a = (-xa**2*yb + xa**2*yc + xb**2*ya - xb**2*yc - xc**2*ya + xc**2*yb - ya**2*yb + ya**2*yc + ya*yb**2 - ya*yc**2 - yb**2*yc + yb*yc**2)/jac
    b = (xa**2*xb - xa**2*xc - xa*xb**2 + xa*xc**2 - xa*yb**2 + xa*yc**2 + xb**2*xc - xb*xc**2 + xb*ya**2 - xb*yc**2 - xc*ya**2 + xc*yb**2)/jac
    c = (-xa**2*xb*yc + xa**2*xc*yb + xa*xb**2*yc - xa*xc**2*yb + xa*yb**2*yc - xa*yb*yc**2 - xb**2*xc*ya + xb*xc**2*ya - xb*ya**2*yc + xb*ya*yc**2 + xc*ya**2*yb - xc*ya*yb**2)/jac
    xo = -a/2
    yo = -b/2
    r = sqrt(xo**2 + yo**2 - c)
    return Circle2D(Vector2D(xo, yo), r)
