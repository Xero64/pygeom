#%%
# Import Dependencies
from typing import TYPE_CHECKING, Union

from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from numpy import asarray, float64, linspace, pi, zeros, tan, sqrt
from pygeom.array3d import zero_arrayvector
from pygeom.geom3d import Vector
from pygeom.tools.bernstein import bernstein_polys

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.array3d import ArrayVector
    Numeric = Union[float64, NDArray[float64]]
    VectorLike = Union[Vector, ArrayVector]

#%%
# Bernstien Polynomial

#%%
# Define the NurbsCurve class
class NurbsSurface():
    points: 'ArrayVector' = None
    weights: 'NDArray[float64]' = None
    degree: int = None

    def __init__(self, points: 'ArrayVector', weights: 'NDArray[float64]') -> None:
        self.points = points
        self.weights = weights
        self.degree = points.size - 1

    def evaluate_at_uv(self, u: 'Numeric', v: 'Numeric') -> 'VectorLike':
        m = self.points.shape[0]
        n = self.points.shape[1]
        if isinstance(u, float64):
            u = asarray([u], dtype=float64)
        if isinstance(v, float64):
            v = asarray([v], dtype=float64)
        u = u.flatten()
        v = v.flatten()
        k = u.size
        l = v.size
        polysu = bernstein_polys(self.degree, u)
        polysv = bernstein_polys(self.degree, v)
        denom = zeros((k, l), dtype=float64)
        points = zero_arrayvector((k, l), dtype=float64)
        for i in range(m):
            polyu = polysu[i, :].reshape((k, 1)).repeat(l, axis=1)
            for j in range(n):
                polyv = polysv[j, :].reshape((1, l)).repeat(k, axis=0)
                weight = self.weights[i, j]
                point = self.points[i, j]
                points += weight*point*polyu*polyv
                denom += weight*polyu*polyv
        points = points / denom
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_points(self, numu: int, numv: int) -> 'ArrayVector':
        u = linspace(0.0, 1.0, numu, dtype=float64)
        v = linspace(0.0, 1.0, numv, dtype=float64)
        return self.evaluate_at_uv(u, v)

num = 21

#%%
# Define the control points and weights
ro = 3.0
ri = 2.0

ctlpts = zero_arrayvector((3, 3))
ctlpts[0, 0] = Vector(ro+ri, 0.0, 0.0)
ctlpts[1, 0] = Vector(ro+ri, ro+ri, 0.0)
ctlpts[2, 0] = Vector(0.0, ro+ri, 0.0)
ctlpts[0, 1] = Vector(ro+ri, 0.0, ri)
ctlpts[1, 1] = Vector(ro+ri, ro+ri, ri)
ctlpts[2, 1] = Vector(0.0, ro+ri, ri)
ctlpts[0, 2] = Vector(ro, 0.0, ri)
ctlpts[1, 2] = Vector(ro, ro, ri)
ctlpts[2, 2] = Vector(0.0, ro, ri)

weights = zeros((3, 3), dtype=float64)
weights[0, 0] = 1.0
weights[1, 0] = 1.0/sqrt(2.0)
weights[2, 0] = 1.0
weights[0, 1] = 1.0/sqrt(2.0)
weights[1, 1] = 1.0/2.0
weights[2, 1] = 1.0/sqrt(2.0)
weights[0, 2] = 1.0
weights[1, 2] = 1.0/sqrt(2.0)
weights[2, 2] = 1.0
# weights[0, 0] = 1.0
# weights[1, 0] = 1.0
# weights[2, 0] = 1.0
# weights[0, 1] = 1.0
# weights[1, 1] = 1.0
# weights[2, 1] = 1.0
# weights[0, 2] = 1.0
# weights[1, 2] = 1.0
# weights[2, 2] = 1.0

print(f'Control Points:\n{ctlpts}\n')
print(f'Weights:\n{weights}\n')

nurbssurface = NurbsSurface(ctlpts, weights)

th = linspace(0.0, pi/2, num, dtype=float64)
u = tan(th/2)
ph = linspace(0.0, pi/2, num, dtype=float64)
v = tan(ph/2)

pnts = nurbssurface.evaluate_at_uv(u, v)

#%%
# Plot the NURBS Surface
fig = figure(figsize=(12, 8))
ax = Axes3D(fig)
fig.add_axes(ax)
ax.view_init(elev=45, azim=45, roll=0)
ax.grid(True)
ax.plot(ctlpts.x, ctlpts.y, ctlpts.z, 'ro', label='Control Points')
ax.plot_wireframe(pnts.x, pnts.y, pnts.z, label='NURBS Surface')
ax.set_box_aspect([ro+ri, ro+ri, ri])
_ = ax.legend()
