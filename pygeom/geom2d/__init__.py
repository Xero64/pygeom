from collections.abc import Iterable

from numpy import cos, sin

from .beziercurve2d import BezierCurve2D as BezierCurve2D
from .beziercurve2d import RationalBezierCurve2D as RationalBezierCurve2D
from .circle2d import Circle2D as Circle2D
from .coordinate2d import Coordinate2D as Coordinate2D
from .cubicspline2d import CubicSpline2D as CubicSpline2D
from .infiniteline2d import InfiniteLine2D as InfiniteLine2D
from .line2d import Line2D as Line2D
from .nurbscurve2d import BSplineCurve2D as BSplineCurve2D
from .nurbscurve2d import NurbsCurve2D as NurbsCurve2D
from .nurbssurface2d import NurbsSurface2D as NurbsSurface2D
from .paramcurve2d import ParamCurve2D as ParamCurve2D
from .paramsurface2d import ParamSurface as ParamSurface2D
from .point2d import Point2D as Point2D
from .tensor2d import Tensor2D as Tensor2D
from .transform2d import Transform2D as Transform2D
from .vector2d import Vector2D as Vector2D

I2D = Vector2D(1.0, 0.0)
J2D = Vector2D(0.0, 1.0)

def midpoint_of_point2ds(pnts: Iterable['Point2D']) -> 'Point2D':
    num = len(pnts)
    x = sum(pnt.x for pnt in pnts)/num
    y = sum(pnt.y for pnt in pnts)/num
    return Point2D(x, y)

def coordinate2d_from_points(pnta: 'Point2D', pntb: 'Point2D') -> 'Coordinate2D':
    """Create a Coordinate2D from two Point2Ds."""
    dirx = pntb - pnta
    return Coordinate2D(pnta, dirx)

def coordinate2d_from_angle(pnt: 'Point2D', angle: float) -> 'Coordinate2D':
    """Create a Coordinate2D from a Point2D and an Angle."""
    dirx = Vector2D(cos(angle), sin(angle))
    return Coordinate2D(pnt, dirx)
