from typing import TYPE_CHECKING, Any, Dict, Union

from numpy import arange, asarray, hstack, vstack, zeros

try:
    from k3d import Plot as Plot
    from k3d import line, lines, mesh, points
    from k3d.objects import Line, Lines, Mesh, Points
except ImportError:
    raise ImportError("k3d is not installed. Please install it using 'pip install k3d'")

if TYPE_CHECKING:
    from ..array2d import NurbsCurve2D
    from ..array3d import NurbsCurve, NurbsSurface
    NurbsCurveLike = Union[NurbsCurve2D, NurbsCurve]
    NurbsSurfaceLike = NurbsSurface
    NurbsLike = Union[NurbsCurveLike, NurbsSurfaceLike]


def k3d_nurbs_curve(curve: 'NurbsCurveLike', **kwargs: Dict[str, Any]) -> Line:
    
    unum = kwargs.get('unum', 12)
    kwargs.setdefault('color', 0xffd500)
    kwargs.setdefault('width', 0.01)

    pnts = curve.evaluate_points(unum)

    if hasattr(pnts, 'z'):
        k3dpnts = pnts.stack_xyz().astype('float32')
    else:
        k3dpnts = hstack((pnts.stack_xy().astype('float32'),
                          zeros((pnts.shape[0], 1), dtype='float32')))
    
    return line(k3dpnts, **kwargs)

def k3d_nurbs_surface(surface: 'NurbsSurfaceLike', **kwargs: Dict[str, Any]) -> Mesh:
    
    unum = kwargs.get('unum', 12)
    vnum = kwargs.get('vnum', 12)
    kwargs.setdefault('color', 0xffd500)
    kwargs.setdefault('wireframe', False)
    kwargs.setdefault('flat_shading', False)

    pnts = surface.evaluate_points(unum, vnum)
    utgts, vtgts = surface.evaluate_tangents(unum, vnum)
    nrms = utgts.cross(vtgts)
    
    num = pnts.size
    ind = arange(pnts.size, dtype=int).reshape(pnts.shape)

    faces = []
    for i in range(pnts.shape[0] - 1):
        for j in range(pnts.shape[1] - 1):
            faces.append([ind[i, j], ind[i+1, j], ind[i+1, j+1]])
            faces.append([ind[i, j], ind[i+1, j+1], ind[i, j+1]])

    faceind = asarray(faces, dtype='uint32')

    pntsxyz = pnts.reshape(num).stack_xyz().astype('float32')
    nrmsxyz = nrms.reshape(num).stack_xyz().astype('float32')
    
    return mesh(pntsxyz, faceind, nrmsxyz, **kwargs)

def k3d_nurbs_control_points(curve: 'NurbsLike', **kwargs: Dict[str, Any]) -> Points:
    
    kwargs.setdefault('color', 0xFF0000)
    scale = kwargs.get('scale', 1.0)
    kwargs.pop('scale', None)
    
    ctlpnts = curve.ctlpnts.flatten()
    weights = curve.weights.flatten()
    
    if hasattr(ctlpnts, 'z'):
        k3dpnts = ctlpnts.stack_xyz().astype('float32')
    else:
        k3dpnts = hstack((ctlpnts.stack_xy().astype('float32'),
                          zeros((ctlpnts.shape[0], 1), dtype='float32')))
    
    kwargs.setdefault('point_sizes', scale*weights.astype('float32'))

    return points(k3dpnts, **kwargs)

def k3d_nurbs_control_polygon(surface: 'NurbsSurfaceLike', **kwargs: Dict[str, Any]) -> Lines:

    kwargs.setdefault('ucolor', 0x00FF00)
    kwargs.setdefault('vcolor', 0x0000FF)
    kwargs['indices_type'] = 'segment'

    inds = arange(surface.ctlpnts.size).reshape(surface.ctlpnts.shape)

    uindsa = inds[:-1, :].reshape((-1, 1))
    uindsb = inds[1:, :].reshape((-1, 1))
    ulinesind = hstack((uindsa, uindsb)).astype('float32')
    uctlpnts = surface.ctlpnts.reshape((-1, 1)).flatten()
    uctlpntsxyz = uctlpnts.stack_xyz().astype('float32')

    vindsa = inds[:, :-1].reshape((-1, 1)) + inds.size
    vindsb = inds[:, 1:].reshape((-1, 1)) + inds.size
    vlinesind = hstack((vindsa, vindsb)).astype('float32')
    vctlpnts = surface.ctlpnts.reshape((-1, 1)).flatten()
    vctlpntsxyz = vctlpnts.stack_xyz().astype('float32')

    ucolors = [kwargs['ucolor']] * inds.size
    vcolors = [kwargs['vcolor']] * inds.size
    uvcolors = ucolors + vcolors
    kwargs['colors'] = uvcolors

    uvctlpntsxyz = vstack((uctlpntsxyz, vctlpntsxyz))
    uvlinesind = vstack((ulinesind, vlinesind))

    return lines(uvctlpntsxyz, uvlinesind, **kwargs)
