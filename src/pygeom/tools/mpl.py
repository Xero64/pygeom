from typing import TYPE_CHECKING, Any

from numpy import ndarray

from ..geom3d import Vector

try:
    from matplotlib.axes import Axes
    from matplotlib.pyplot import figure
except ImportError:
    raise ImportError("k3d is not installed. Please install it using 'pip install k3d'")

if TYPE_CHECKING:
    from ..geom2d import (CubicSpline2D, NurbsCurve2D, NurbsSurface2D,
                          ParamCurve2D)
    from ..geom3d import (CubicSpline, NurbsCurve, NurbsSurface, ParamCurve,
                          ParamSurface)
    CurveLike = NurbsCurve2D | NurbsCurve | ParamCurve2D | ParamCurve | CubicSpline2D | CubicSpline
    SurfaceLike = NurbsSurface2D | NurbsSurface | ParamSurface
    NurbsLike = NurbsCurve2D | NurbsCurve | NurbsSurface

NUM = 36

def plot_curve(curve: 'CurveLike', **kwargs: dict[str, Any]) -> Axes:

    ax = kwargs.pop('ax', None)
    num = kwargs.pop('num', NUM)
    t = kwargs.pop('t', curve.evaluate_t(num))

    pnts = curve.evaluate_points_at_t(t)

    if ax is None:
        figsize = kwargs.pop('figsize', (10, 8))
        fig = figure(figsize=figsize)
        if hasattr(pnts, 'z'):
            ax: Axes = fig.add_subplot(projection='3d')
        else:
            ax = fig.gca()
        ax.set_aspect('equal')
        ax.grid(True)

    if hasattr(pnts, 'z'):
        ax.plot(pnts.x, pnts.y, pnts.z, **kwargs)
    else:
        ax.plot(pnts.x, pnts.y, **kwargs)

    return ax

def quiver_tangents(curve: 'CurveLike', **kwargs: dict[str, Any]) -> Axes:

    ax = kwargs.pop('ax', None)
    num = kwargs.pop('num', NUM)
    t = kwargs.pop('t', curve.evaluate_t(num))
    scale = kwargs.pop('scale', 1.0)

    pnts = curve.evaluate_points_at_t(t)
    tgts = curve.evaluate_first_derivatives_at_t(t).to_unit()

    if ax is None:
        figsize = kwargs.pop('figsize', (10, 8))
        fig = figure(figsize=figsize)
        if hasattr(pnts, 'z'):
            ax: Axes = fig.add_subplot(projection='3d')
        else:
            ax = fig.gca()
        ax.set_aspect('equal')
        ax.grid(True)

    if hasattr(pnts, 'z'):
        ax.quiver(pnts.x, pnts.y, pnts.z,
                  tgts.x*scale, tgts.y*scale, tgts.z*scale, **kwargs)
    else:
        ax.quiver(pnts.x, pnts.y, tgts.x*scale, tgts.y*scale, **kwargs)

    return ax

def quiver_normals(curve: 'CurveLike', **kwargs: dict[str, Any]) -> Axes:

    ax = kwargs.pop('ax', None)
    num = kwargs.pop('num', NUM)
    t = kwargs.pop('t', curve.evaluate_t(num))
    scale = kwargs.pop('scale', 1.0)

    pnts = curve.evaluate_points_at_t(t)
    nrms = curve.evaluate_second_derivatives_at_t(t).to_unit()

    if ax is None:
        figsize = kwargs.pop('figsize', (10, 8))
        fig = figure(figsize=figsize)
        if hasattr(pnts, 'z'):
            ax: Axes = fig.add_subplot(projection='3d')
        else:
            ax = fig.gca()
        ax.grid(True)

    if hasattr(pnts, 'z'):
        ax.quiver(pnts.x, pnts.y, pnts.z,
                  nrms.x*scale, nrms.y*scale, nrms.z*scale, **kwargs)
    else:
        ax.quiver(pnts.x, pnts.y, nrms.x*scale, nrms.y*scale, **kwargs)

    return ax

def plot_points(curve: 'CurveLike', **kwargs: dict[str, Any]) -> Axes:

    ax = kwargs.pop('ax', None)
    num = kwargs.pop('num', NUM)
    t = kwargs.pop('t', curve.evaluate_t(num))

    pnts = curve.evaluate_points_at_t(t)

    if ax is None:
        figsize = kwargs.pop('figsize', (10, 8))
        fig = figure(figsize=figsize)
        ax = fig.gca()
        ax.grid(True)

    kwargs_x = kwargs.copy()
    kwargs_x['label'] = f'{kwargs_x.get("label", "Curve")} X'
    ax.plot(t, pnts.x, **kwargs_x)

    kwargs_y = kwargs.copy()
    kwargs_y['label'] = f'{kwargs_y.get("label", "Curve")} Y'
    ax.plot(t, pnts.y, **kwargs_y)

    if hasattr(pnts, 'z'):
        kwargs_z = kwargs.copy()
        kwargs_z['label'] = f'{kwargs_z.get("label", "Curve")} Z'
        ax.plot(t, pnts.z, **kwargs_z)

    return ax

def plot_first_derivatives(curve: 'CurveLike', **kwargs: dict[str, Any]) -> Axes:

    ax = kwargs.pop('ax', None)
    num = kwargs.pop('num', NUM)
    t = kwargs.pop('t', curve.evaluate_t(num))

    deriv1 = curve.evaluate_first_derivatives_at_t(t)

    if ax is None:
        figsize = kwargs.pop('figsize', (10, 8))
        fig = figure(figsize=figsize)
        ax = fig.gca()
        ax.grid(True)

    kwargs_x = kwargs.copy()
    kwargs_x['label'] = f'{kwargs_x.get("label", "Curve")} dXdt'
    ax.plot(t, deriv1.x, **kwargs_x)

    kwargs_y = kwargs.copy()
    kwargs_y['label'] = f'{kwargs_y.get("label", "Curve")} dYdt'
    ax.plot(t, deriv1.y, **kwargs_y)

    if hasattr(deriv1, 'z'):
        kwargs_z = kwargs.copy()
        kwargs_z['label'] = f'{kwargs_z.get("label", "Curve")} dZdt'
        ax.plot(t, deriv1.z, **kwargs_z)

    return ax

def plot_second_derivatives(curve: 'CurveLike', **kwargs: dict[str, Any]) -> Axes:

    ax = kwargs.pop('ax', None)
    num = kwargs.pop('num', NUM)
    t = kwargs.pop('t', curve.evaluate_t(num))

    deriv2 = curve.evaluate_second_derivatives_at_t(t)

    if ax is None:
        figsize = kwargs.pop('figsize', (10, 8))
        fig = figure(figsize=figsize)
        ax = fig.gca()
        ax.grid(True)

    kwargs_x = kwargs.copy()
    kwargs_x['label'] = f'{kwargs_x.get("label", "Curve")} d2Xdt2'
    ax.plot(t, deriv2.x, **kwargs_x)

    kwargs_y = kwargs.copy()
    kwargs_y['label'] = f'{kwargs_y.get("label", "Curve")} d2Ydt2'
    ax.plot(t, deriv2.y, **kwargs_y)

    if hasattr(deriv2, 'z'):
        kwargs_z = kwargs.copy()
        kwargs_z['label'] = f'{kwargs_z.get("label", "Curve")} d2Zdt2'
        ax.plot(t, deriv2.z, **kwargs_z)

    return ax

def plot_curvature(curve: 'CurveLike', **kwargs: dict[str, Any]) -> Axes:

    ax = kwargs.pop('ax', None)
    num = kwargs.pop('num', NUM)
    t = kwargs.pop('t', curve.evaluate_t(num))

    curvature = curve.evaluate_curvatures_at_t(t)

    if ax is None:
        figsize = kwargs.pop('figsize', (10, 8))
        fig = figure(figsize=figsize)
        ax = fig.gca()
        ax.grid(True)

    if isinstance(curvature, Vector):

        kwargs_x = kwargs.copy()
        kwargs_x['label'] = f'{kwargs_x.get("label", "Curve")} Curvature X'
        ax.plot(t, curvature.x, **kwargs_x)

        kwargs_y = kwargs.copy()
        kwargs_y['label'] = f'{kwargs_y.get("label", "Curve")} Curvature Y'
        ax.plot(t, curvature.y, **kwargs_y)

        kwargs_z = kwargs.copy()
        kwargs_z['label'] = f'{kwargs_z.get("label", "Curve")} Curvature Z'
        ax.plot(t, curvature.z, **kwargs_z)

    elif isinstance(curvature, ndarray):

        ax.plot(t, curvature, **kwargs)

    else:

        raise ValueError(f'Unknown curvature type: {type(curvature)}')

    return ax
