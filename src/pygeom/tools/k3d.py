from typing import TYPE_CHECKING, Any

from numpy import arange, asarray, hstack, vstack, zeros

from ..geom2d import Vector2D
from ..geom3d import Vector

try:
    from k3d import Plot as Plot
    from k3d import line, lines, mesh, points, vectors
    from k3d.objects import Line, Lines, Mesh, Points, Vectors
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


def make_vector_3d(vec: Vector | Vector2D) -> Vector:
    if hasattr(vec, 'z'):
        return vec
    else:
        return Vector(vec.x, vec.y, zeros(vec.shape))

def k3d_curve(curve: 'CurveLike', **kwargs: dict[str, Any]) -> Line:

    num = kwargs.get('num', 12)
    kwargs.setdefault('color', 0xffd500)
    kwargs.setdefault('width', 0.01)

    pnts = make_vector_3d(curve.evaluate_points(num))

    k3dpnts = pnts.stack_xyz().astype('float32')

    return line(k3dpnts, **kwargs)

def k3d_curve_tangents(curve: 'CurveLike', **kwargs: dict[str, Any]) -> Vectors:

    num = kwargs.get('num', 12)
    scale = kwargs.pop('scale', 1.0)

    kwargs.setdefault('color', 0x00ff00)

    pnts = make_vector_3d(curve.evaluate_points(num))
    tgts = make_vector_3d(curve.evaluate_first_derivatives(num).to_unit())

    k3dpnts = pnts.stack_xyz().astype('float32')
    k3dtgts = tgts.stack_xyz().astype('float32')*scale

    return vectors(k3dpnts, k3dtgts, **kwargs)

def k3d_curve_normals(curve: 'CurveLike', **kwargs: dict[str, Any]) -> Vectors:

    num = kwargs.get('num', 12)
    scale = kwargs.pop('scale', 1.0)

    kwargs.setdefault('color', 0xff0000)

    pnts = make_vector_3d(curve.evaluate_points(num))
    nrms = make_vector_3d(curve.evaluate_second_derivatives(num).to_unit())

    k3dpnts = pnts.stack_xyz().astype('float32')
    k3dnrms = nrms.stack_xyz().astype('float32')*scale

    return vectors(k3dpnts, k3dnrms, **kwargs)

def k3d_curve_binormals(curve: 'CurveLike', **kwargs: dict[str, Any]) -> Vectors:

    num = kwargs.get('num', 12)
    scale = kwargs.pop('scale', 1.0)

    kwargs.setdefault('color', 0x0000ff)

    pnts = make_vector_3d(curve.evaluate_points(num))
    tgts = make_vector_3d(curve.evaluate_first_derivatives(num).to_unit())
    nrms = make_vector_3d(curve.evaluate_second_derivatives(num).to_unit())
    bins = tgts.cross(nrms).to_unit()

    k3dpnts = pnts.stack_xyz().astype('float32')
    binsxyz = bins.stack_xyz().astype('float32')*scale

    return vectors(k3dpnts, binsxyz, **kwargs)

def k3d_surface(surface: 'SurfaceLike', **kwargs: dict[str, Any]) -> Mesh:

    unum = kwargs.pop('unum', 12)
    vnum = kwargs.pop('vnum', 12)
    kwargs.setdefault('color', 0xffd500)
    kwargs.setdefault('wireframe', False)
    kwargs.setdefault('flat_shading', False)

    u, v = surface.evaluate_uv(unum, vnum)
    u = kwargs.pop('u', u)
    v = kwargs.pop('v', v)

    pnts = make_vector_3d(surface.evaluate_points_at_uv(u, v))
    utgts, vtgts = surface.evaluate_tangents_at_uv(u, v)
    utgts = make_vector_3d(utgts)
    vtgts = make_vector_3d(vtgts)
    nrms = utgts.cross(vtgts).to_unit()

    num = pnts.size
    ind = arange(pnts.size, dtype=int).reshape(pnts.shape)

    faces = []
    for i in range(pnts.shape[0] - 1):
        for j in range(pnts.shape[1] - 1):
            faces.append([ind[i, j], ind[i+1, j], ind[i+1, j+1]])
            faces.append([ind[i, j], ind[i+1, j+1], ind[i, j+1]])

    faceind = asarray(faces, dtype='uint32')

    k3dpnts = pnts.reshape(num).stack_xyz().astype('float32')
    k3dnrms = nrms.reshape(num).stack_xyz().astype('float32')

    return mesh(k3dpnts, faceind, k3dnrms, **kwargs)

def k3d_surface_normals(surface: 'SurfaceLike', **kwargs: dict[str, Any]) -> Vectors:

    unum = kwargs.pop('unum', 12)
    vnum = kwargs.pop('vnum', 12)
    scale = kwargs.pop('scale', 1.0)
    kwargs.setdefault('color', 0xff0000)
    kwargs.setdefault('head_size', 0.1)
    kwargs.setdefault('line_width', 0.01)

    u, v = surface.evaluate_uv(unum, vnum)
    u = kwargs.pop('u', u)
    v = kwargs.pop('v', v)

    pnts = make_vector_3d(surface.evaluate_points_at_uv(u, v))
    utgts, vtgts = surface.evaluate_tangents_at_uv(u, v)
    utgts = make_vector_3d(utgts)
    vtgts = make_vector_3d(vtgts)
    nrms = utgts.cross(vtgts).to_unit()

    num = pnts.size

    k3dpnts = pnts.reshape(num).stack_xyz().astype('float32')
    k3dnrms = nrms.reshape(num).stack_xyz().astype('float32')*scale

    return vectors(k3dpnts, k3dnrms, **kwargs)

def k3d_surface_tangents(surface: 'SurfaceLike',
                         **kwargs: dict[str, Any]) -> tuple[Vectors, Vectors]:

    unum = kwargs.pop('unum', 12)
    vnum = kwargs.pop('vnum', 12)
    scale = kwargs.pop('scale', 1.0)
    kwargs.setdefault('head_size', 0.1)
    kwargs.setdefault('line_width', 0.01)

    u, v = surface.evaluate_uv(unum, vnum)
    u = kwargs.pop('u', u)
    v = kwargs.pop('v', v)

    kwargsu = kwargs.copy()
    kwargsu.setdefault('color', 0x00ff00)

    kwargsv = kwargs.copy()
    kwargsv.setdefault('color', 0x0000ff)

    pnts = make_vector_3d(surface.evaluate_points_at_uv(u, v))
    tgtsu, tgtsv = surface.evaluate_tangents_at_uv(u, v)
    tgtsu = make_vector_3d(tgtsu.to_unit())
    tgtsv = make_vector_3d(tgtsv.to_unit())

    num = pnts.size

    k3dpnts = pnts.reshape(num).stack_xyz().astype('float32')
    k3dtgtsu = tgtsu.reshape(num).stack_xyz().astype('float32')*scale
    k3dtgtsv = tgtsv.reshape(num).stack_xyz().astype('float32')*scale

    return vectors(k3dpnts, k3dtgtsu, **kwargsu), \
           vectors(k3dpnts, k3dtgtsv, **kwargsv)

def k3d_nurbs_control_points(curve: 'NurbsLike', **kwargs: dict[str, Any]) -> Points:

    kwargs.setdefault('color', 0xFF0000)
    scale = kwargs.get('scale', 1.0)
    kwargs.pop('scale', None)

    ctlpnts = make_vector_3d(curve.ctlpnts.ravel())
    weights = curve.weights.ravel()

    k3dpnts = ctlpnts.stack_xyz().astype('float32')

    kwargs.setdefault('point_sizes', scale*weights.astype('float32'))

    return points(k3dpnts, **kwargs)

def k3d_nurbs_control_polygon(surface: 'NurbsSurface', **kwargs: dict[str, Any]) -> Lines:

    kwargs.setdefault('ucolor', 0x00FF00)
    kwargs.setdefault('vcolor', 0x0000FF)
    kwargs['indices_type'] = 'segment'

    inds = arange(surface.ctlpnts.size).reshape(surface.ctlpnts.shape)

    uindsa = inds[:-1, :].reshape((-1, 1))
    uindsb = inds[1:, :].reshape((-1, 1))
    ulinesind = hstack((uindsa, uindsb)).astype('float32')
    uctlpnts = make_vector_3d(surface.ctlpnts.reshape((-1, 1)).ravel())
    uctlk3dpnts = uctlpnts.stack_xyz().astype('float32')

    vindsa = inds[:, :-1].reshape((-1, 1)) + inds.size
    vindsb = inds[:, 1:].reshape((-1, 1)) + inds.size
    vlinesind = hstack((vindsa, vindsb)).astype('float32')
    vctlpnts = make_vector_3d(surface.ctlpnts.reshape((-1, 1)).ravel())
    vctlk3dpnts = vctlpnts.stack_xyz().astype('float32')

    ucolors = [kwargs['ucolor']] * inds.size
    vcolors = [kwargs['vcolor']] * inds.size
    uvcolors = ucolors + vcolors
    kwargs['colors'] = uvcolors

    uvctlk3dpnts = vstack((uctlk3dpnts, vctlk3dpnts))
    uvlinesind = vstack((ulinesind, vlinesind))

    return lines(uvctlk3dpnts, uvlinesind, **kwargs)
