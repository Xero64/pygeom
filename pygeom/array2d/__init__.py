from typing import TYPE_CHECKING

from numpy import allclose, float64, hsplit, hstack
from numpy.linalg import solve

from .arraytensor2d import ArrayTensor2D as ArrayTensor2D
from .arraytensor2d import zero_arraytensor2d as zero_arraytensor2d
from .arrayvector2d import ArrayVector2D as ArrayVector2D
from .arrayvector2d import zero_arrayvector2d as zero_arrayvector2d
from .beziercurve2d import BezierCurve2D as BezierCurve2D
from .beziercurve2d import RationalBezierCurve2D as RationalBezierCurve2D
from .nurbscurve2d import BSplineCurve2D as BSplineCurve2D
from .nurbscurve2d import NurbsCurve2D as NurbsCurve2D
from .paramcurve2d import ParamCurve2D as ParamCurve2D

if TYPE_CHECKING:
    from numpy.typing import NDArray

def solve_arrayvector2d(a: 'NDArray[float64]', b: 'ArrayVector2D') -> 'ArrayVector2D':
    newb = hstack(b.to_xy())
    newc = solve(a, newb)
    x, y = hsplit(newc, 2)
    return ArrayVector2D(x, y)

def arrayvector2d_allclose(a: ArrayVector2D, b: ArrayVector2D,
                           rtol: float=1e-09, atol: float=0.0) -> bool:
    """Returns True if two ArrayVector2Ds are close enough to be considered equal."""
    return allclose(a.x, b.x, rtol=rtol, atol=atol) and \
        allclose(a.y, b.y, rtol=rtol, atol=atol)

def matmul_arrayvector2d(a: 'NDArray[float64]', b: 'ArrayVector2D') -> 'ArrayVector2D':
    """Returns the matrix multiplication of a and b."""
    x = a @ b.x
    y = a @ b.y
    return ArrayVector2D(x, y)
